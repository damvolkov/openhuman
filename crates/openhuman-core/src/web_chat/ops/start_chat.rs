//! The primary and parallel turn entry point: validate the submission, hand it
//! to the turn runner, and answer how it was received.
//!
//! Everything about the turn itself (the ingress policy, queueing against the
//! thread's in-flight turn, guards, cancellation, delivery) is the runner and
//! its ports (`runner.rs`). This file builds the request and learns, from the
//! first of the runner's answers, whether the message started a turn, joined
//! the running turn's queue, was an approval reply, or was refused.

use std::time::Duration;

use tinyagents_harness::turn_runner::{QueueDisposition, TurnRequest, TurnRunnerError};
use tinychannels_bus::web::{ChatRequestMetadata, QueueMode};
use uuid::Uuid;

use super::super::runner::{runtime, scope_of, Admission};
use super::super::turn_activity::configured_idle_window;
use crate::agent::queued_turn::QueuedPayload;
use crate::core::events::DomainEvent;
use crate::security::prompt_injection::PromptEnforcementAction;

/// `start_chat`'s error type.
///
/// `Guardrail` is a structured verdict from the prompt-injection/security
/// guardrail (`security::prompt_injection::enforce_prompt_input`) — the
/// frontend classifies on this variant (`chat_error.error_type == "guardrail"`
/// and a `guardrail` payload) instead of pattern-matching the user-facing
/// message string. Every other rejection (validation, a configured
/// `beforeSubmitPrompt` hook block, an approval-routing failure) stays
/// `Other`, which `Display`s exactly like the plain `String` errors this
/// replaced — existing `.to_string()` / `{err}` call sites need no other
/// change.
#[derive(Debug, Clone)]
pub enum StartChatError {
    Guardrail {
        verdict: String,
        score: f64,
        reasons: Vec<crate::web_chat::GuardrailReason>,
    },
    Other(String),
}

impl std::fmt::Display for StartChatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            // The same user-facing copy `prompt_guard_user_message` gives a
            // fresh rejection — a caller that only has `.to_string()` (a
            // plain-string RPC error, a `{err}` log line) still gets an
            // actionable message, not a bare verdict/score dump.
            StartChatError::Guardrail { verdict, .. } => {
                f.write_str(guardrail_verdict_user_message(verdict))
            }
            StartChatError::Other(message) => write!(f, "{message}"),
        }
    }
}

/// User-facing copy for a guardrail verdict string (`"block"` /
/// `"review_blocked"` / `"allow"` — see the `match` in [`start_chat`] that
/// builds [`StartChatError::Guardrail`]). Shared by `Display` above so a
/// plain-string consumer of the error still reads the same rejection copy
/// [`prompt_guard_user_message`] gives a fresh (non-error-wrapped) decision.
fn guardrail_verdict_user_message(verdict: &str) -> &'static str {
    match verdict {
        "block" => prompt_guard_user_message(PromptEnforcementAction::Blocked),
        "review_blocked" => prompt_guard_user_message(PromptEnforcementAction::ReviewBlocked),
        _ => prompt_guard_user_message(PromptEnforcementAction::Allow),
    }
}

fn submission_has_content(
    message: &str,
    stored_attachments: Option<&crate::agent::queued_turn::StoredAttachmentInput>,
    attachment_uploads: Option<&[tinychannels_bus::ContentPart]>,
) -> bool {
    !message.trim().is_empty()
        || stored_attachments.is_some_and(|input| !input.parts.is_empty())
        || attachment_uploads.is_some_and(|parts| !parts.is_empty())
}

impl std::error::Error for StartChatError {}

impl From<String> for StartChatError {
    fn from(message: String) -> Self {
        StartChatError::Other(message)
    }
}

impl From<&str> for StartChatError {
    fn from(message: &str) -> Self {
        StartChatError::Other(message.to_string())
    }
}

/// Sentinel prefix a JSON-RPC/string-error caller can match on to recover the
/// structured guardrail verdict, the same pattern as
/// `core::observability::BACKEND_UNAVAILABLE_PREFIX`: the RPC surface only
/// carries `Result<_, String>`, so the socket path's `chat_error.guardrail`
/// payload gets a string-shaped equivalent here rather than a second, looser
/// error shape.
pub const GUARDRAIL_ERROR_PREFIX: &str = "GUARDRAIL:";

impl From<StartChatError> for String {
    fn from(error: StartChatError) -> Self {
        match error {
            StartChatError::Guardrail {
                verdict,
                score,
                reasons,
            } => {
                let payload = crate::web_chat::GuardrailPayload {
                    verdict,
                    score,
                    reasons,
                };
                format!(
                    "{GUARDRAIL_ERROR_PREFIX}{}",
                    serde_json::to_string(&payload).unwrap_or_default()
                )
            }
            StartChatError::Other(message) => message,
        }
    }
}

fn prompt_guard_user_message(action: PromptEnforcementAction) -> &'static str {
    match action {
        PromptEnforcementAction::Allow => "Message accepted.",
        PromptEnforcementAction::Blocked => {
            "Your message was blocked by a security policy. Please rephrase and remove instruction-override or secret-exfiltration requests."
        }
        PromptEnforcementAction::ReviewBlocked => {
            "Your message was flagged for security review and was not processed. Please rephrase the request in a direct, task-focused way."
        }
    }
}

/// Resolve the per-turn wall-clock backstop. Returns `None` when disabled
/// (env `OPENHUMAN_WEB_TURN_TIMEOUT_SECS=0`).
///
/// This is the turn's fixed total ceiling (issue #4746), and the single source
/// of truth for the turn's deadlines (`web_chat::turn_guards::backstop`,
/// default 3600s, the harness's own turn ceiling). The harness derives two
/// earlier points from it: a **wind-down** at backstop minus
/// `min(120s, backstop/5)`, where it pauses the run at its next safe point so
/// the session driver writes the grounded close from the work already done, and
/// a **hard stop** at backstop minus `min(60s, backstop/10)`.
///
/// It is not the hang detector. That is the runner's idle guard, which ends a
/// turn that has been silent with no model or tool call in flight. Override via
/// `OPENHUMAN_WEB_TURN_TIMEOUT_SECS`; set it to `0` to disable the backstop and
/// the derived deadlines with it.
fn web_turn_deadline() -> Option<Duration> {
    crate::web_chat::turn_guards::backstop::configured_backstop()
}

/// The backstop for a turn that has to be allowed to run long: one on a local /
/// self-hosted provider (#6042), whose prefill alone can take many minutes. The
/// explicit `OPENHUMAN_WEB_TURN_TIMEOUT_SECS` override still wins; otherwise the
/// default is the local turn ceiling plus a grace period, never the hosted
/// backstop, which sits below the local harness ceilings and would pre-empt them.
async fn resolve_web_turn_deadline(ctx: &crate::core::runtime::DispatchCtx) -> Option<Duration> {
    if std::env::var("OPENHUMAN_WEB_TURN_TIMEOUT_SECS").is_ok() {
        return web_turn_deadline();
    }
    let local = match ctx.load_config().await {
        Ok(config) => crate::agent::tinyagents::chat_provider_is_local(&config),
        Err(error) => {
            log::debug!("[web-channel] backstop locality probe skipped: {error}");
            false
        }
    };
    if local {
        let secs = crate::agent::tinyagents::local_web_turn_backstop_secs();
        log::debug!("[web-channel] local provider: turn backstop {secs:?}s (#6042)");
        return secs.map(Duration::from_secs);
    }
    web_turn_deadline()
}

fn parse_mode(mode: Option<&str>) -> QueueMode {
    match mode {
        Some("steer") => QueueMode::Steer,
        Some("followup") => QueueMode::Followup,
        Some("collect") => QueueMode::Collect,
        Some("parallel") => QueueMode::Parallel,
        _ => QueueMode::Interrupt,
    }
}

fn validate_queue_mode_media(
    mode: QueueMode,
    has_voice_note: bool,
    has_stored_media: bool,
    has_uploads: bool,
) -> Result<(), StartChatError> {
    if matches!(mode, QueueMode::Collect | QueueMode::Steer)
        && (has_voice_note || has_stored_media || has_uploads)
    {
        return Err(StartChatError::Other(
            "collect and steer queue modes do not support typed attachments".to_string(),
        ));
    }
    Ok(())
}

fn disposition(mode: QueueMode) -> QueueDisposition {
    match mode {
        QueueMode::Interrupt => QueueDisposition::Interrupt,
        QueueMode::Steer => QueueDisposition::Steer,
        QueueMode::Followup => QueueDisposition::Followup,
        QueueMode::Collect => QueueDisposition::Collect,
        QueueMode::Parallel => QueueDisposition::Parallel,
    }
}

/// Returned boxed and `#[inline(never)]` on purpose: other crates await this
/// (`openhuman-rpc`, `openhuman-embed`), and an `async fn` body is otherwise
/// re-instantiated inside every calling crate's state machine. Boxing here
/// keeps one copy, compiled in this crate.
#[allow(clippy::too_many_arguments)]
#[inline(never)]
pub fn start_chat<'a>(
    ctx: &'a crate::core::runtime::DispatchCtx,
    client_id: &'a str,
    thread_id: &'a str,
    message: &'a str,
    model_override: Option<String>,
    temperature: Option<f64>,
    locale: Option<String>,
    queue_mode: Option<String>,
    metadata: ChatRequestMetadata,
) -> futures::future::BoxFuture<'a, Result<String, StartChatError>> {
    start_chat_with_voice_note(
        ctx,
        client_id,
        thread_id,
        message,
        model_override,
        temperature,
        locale,
        queue_mode,
        metadata,
        None,
    )
}

pub(super) fn start_chat_with_voice_note<'a>(
    ctx: &'a crate::core::runtime::DispatchCtx,
    client_id: &'a str,
    thread_id: &'a str,
    message: &'a str,
    model_override: Option<String>,
    temperature: Option<f64>,
    locale: Option<String>,
    queue_mode: Option<String>,
    metadata: ChatRequestMetadata,
    voice_note_id: Option<String>,
) -> futures::future::BoxFuture<'a, Result<String, StartChatError>> {
    start_chat_with_stored_attachments(
        ctx,
        client_id,
        thread_id,
        message,
        model_override,
        temperature,
        locale,
        queue_mode,
        metadata,
        voice_note_id,
        None,
    )
}

pub(super) fn start_chat_with_stored_attachments<'a>(
    ctx: &'a crate::core::runtime::DispatchCtx,
    client_id: &'a str,
    thread_id: &'a str,
    message: &'a str,
    model_override: Option<String>,
    temperature: Option<f64>,
    locale: Option<String>,
    queue_mode: Option<String>,
    metadata: ChatRequestMetadata,
    voice_note_id: Option<String>,
    stored_attachments: Option<crate::agent::queued_turn::StoredAttachmentInput>,
) -> futures::future::BoxFuture<'a, Result<String, StartChatError>> {
    start_chat_with_attachments(
        ctx,
        client_id,
        thread_id,
        message,
        model_override,
        temperature,
        locale,
        queue_mode,
        metadata,
        voice_note_id,
        stored_attachments,
        None,
    )
}

pub(super) fn start_chat_with_attachments<'a>(
    ctx: &'a crate::core::runtime::DispatchCtx,
    client_id: &'a str,
    thread_id: &'a str,
    message: &'a str,
    model_override: Option<String>,
    temperature: Option<f64>,
    locale: Option<String>,
    queue_mode: Option<String>,
    metadata: ChatRequestMetadata,
    voice_note_id: Option<String>,
    stored_attachments: Option<crate::agent::queued_turn::StoredAttachmentInput>,
    attachment_uploads: Option<Vec<tinychannels_bus::ContentPart>>,
) -> futures::future::BoxFuture<'a, Result<String, StartChatError>> {
    Box::pin(start_chat_inner(
        ctx,
        client_id,
        thread_id,
        message,
        model_override,
        temperature,
        locale,
        queue_mode,
        metadata,
        voice_note_id,
        stored_attachments,
        attachment_uploads,
    ))
}

#[allow(clippy::too_many_arguments)]
async fn start_chat_inner(
    ctx: &crate::core::runtime::DispatchCtx,
    client_id: &str,
    thread_id: &str,
    message: &str,
    model_override: Option<String>,
    temperature: Option<f64>,
    locale: Option<String>,
    queue_mode: Option<String>,
    metadata: ChatRequestMetadata,
    voice_note_id: Option<String>,
    stored_attachments: Option<crate::agent::queued_turn::StoredAttachmentInput>,
    attachment_uploads: Option<Vec<tinychannels_bus::ContentPart>>,
) -> Result<String, StartChatError> {
    let client_id = client_id.trim().to_string();
    let thread_id = thread_id.trim().to_string();
    let message = message.trim().to_string();

    if client_id.is_empty() {
        return Err(StartChatError::Other("client_id is required".to_string()));
    }
    if thread_id.is_empty() {
        return Err(StartChatError::Other("thread_id is required".to_string()));
    }
    if !submission_has_content(
        &message,
        stored_attachments.as_ref(),
        attachment_uploads.as_deref(),
    ) {
        return Err(StartChatError::Other("message is required".to_string()));
    }
    let mode = parse_mode(queue_mode.as_deref());
    validate_queue_mode_media(
        mode,
        voice_note_id.is_some(),
        stored_attachments
            .as_ref()
            .is_some_and(|input| !input.parts.is_empty()),
        attachment_uploads
            .as_ref()
            .is_some_and(|parts| !parts.is_empty()),
    )?;
    // A profile this node no longer hosts (its lease was lost or released)
    // starts no new turn here. No-op outside SaaS.
    crate::profiles::host::ensure_hosted(ctx).map_err(StartChatError::Other)?;

    let pending_attachments = match attachment_uploads {
        Some(parts) if !parts.is_empty() => Some(
            crate::threads::attachments::stage_queued_parts(ctx, &thread_id, &parts)
                .await
                .map(|parts| crate::agent::queued_turn::PendingAttachmentInput { parts })
                .map_err(StartChatError::Other)?,
        ),
        _ => None,
    };

    let request_id = Uuid::new_v4().to_string();
    if let Some(note_id) = voice_note_id.as_deref() {
        bind_voice_note_request_id(ctx, &thread_id, &message, note_id, &request_id).await?;
    }
    let preview = crate::agent::queued_turn::text_preview(&message);
    let scope = scope_of(ctx);
    let queued_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let request = TurnRequest {
        scope: scope.clone(),
        thread_id: thread_id.clone(),
        request_id: request_id.clone(),
        text: message,
        payload: QueuedPayload {
            ctx: Some(ctx.clone()),
            client_id: client_id.clone(),
            trusted_turn: None,
            model_override,
            temperature,
            locale,
            metadata,
            queued_at_ms,
            deferred_user_message: mode == QueueMode::Followup && stored_attachments.is_none(),
            stored_attachments,
            pending_attachments: pending_attachments.clone(),
            native_message_input: None,
            native_staged_parts: Vec::new(),
            native_staged_lease: None,
            native_message_id: None,
            native_user_text: None,
            native_preparation_error: None,
        },
        host_authored: false,
        disposition: disposition(mode),
        deadline: resolve_web_turn_deadline(ctx).await,
        idle_timeout: Some(configured_idle_window()),
    };

    let rt = runtime(ctx);
    let (admission_tx, admission_rx) = tokio::sync::oneshot::channel();
    rt.shared
        .with(&request_id, |slot| slot.admission = Some(admission_tx));
    if mode == QueueMode::Parallel {
        log::info!(
            "[web-channel] starting PARALLEL forked turn thread_id={thread_id} request_id={request_id}"
        );
    }
    let (runner, shared) = (rt.runner.clone(), rt.shared.clone());
    let id = request_id.clone();
    ctx.spawn(async move {
        let outcome = runner.run(request).await;
        // The first answer wins: a turn that started already told its
        // submitter, and the refusals told theirs at the door.
        match &outcome {
            Err(TurnRunnerError::Queued) => {
                shared.admit(&id, Admission::Queued);
                shared.mark_queued(&id);
            }
            Err(TurnRunnerError::Rejected(reason) | TurnRunnerError::Failed(reason)) => {
                shared.admit(
                    &id,
                    Admission::Rejected(StartChatError::Other(reason.clone())),
                );
            }
            Err(TurnRunnerError::Consumed | TurnRunnerError::Cancelled) | Ok(_) => {
                shared.admit(&id, Admission::Started);
            }
        }
        log::debug!(
            "[web-channel] turn finished request_id={id} ok={}",
            outcome.is_ok()
        );
    });

    match admission_rx.await.unwrap_or(Admission::Started) {
        Admission::Started => Ok(request_id),
        Admission::Consumed(approval_id) => {
            rt.shared.take(&request_id);
            Ok(approval_id)
        }
        Admission::Rejected(error) => {
            rt.shared.take(&request_id);
            Err(error)
        }
        Admission::Queued => {
            let queue_depth = rt.runner.queue_status(&scope, &thread_id).await.total;
            log::info!(
                "[web-channel] queued {mode} message thread_id={thread_id} request_id={request_id} queue_depth={queue_depth}"
            );
            crate::core::bus::BUS.publish(DomainEvent::RunQueueMessageQueued {
                thread_id: thread_id.clone(),
                mode: mode.to_string(),
                queue_depth,
                item_id: Some(request_id.clone()),
                text_preview: Some(preview),
            });
            serde_json::to_string(&crate::threads::rpc_models::QueuedChatAck {
                queued: true,
                queue_mode: mode.to_string(),
                client_id,
                thread_id,
                request_id,
                queue_depth,
                queued_attachments: pending_attachments
                    .map(|input| input.parts)
                    .unwrap_or_default(),
            })
            .map_err(|error| StartChatError::Other(format!("serialize queued chat ack: {error}")))
        }
    }
}

/// Store the host-generated turn id on the one durable user row named by the
/// typed note reference before the runner can consume the request. This keeps
/// transcript projection unambiguous for repeated identical transcriptions.
async fn bind_voice_note_request_id(
    ctx: &crate::core::runtime::DispatchCtx,
    thread_id: &str,
    message: &str,
    note_id: &str,
    request_id: &str,
) -> Result<(), StartChatError> {
    use crate::threads::store::{blocking, ConversationMessagePatch};

    let workspace_dir = ctx
        .load_config()
        .await
        .map_err(|error| StartChatError::Other(format!("load voice note workspace: {error}")))?
        .workspace_dir;
    let rows = blocking::get_messages(ctx.clone(), workspace_dir.clone(), thread_id.to_string())
        .await
        .map_err(StartChatError::Other)?;
    let matches = rows
        .iter()
        .filter(|row| row.sender == "user" && row.content == message)
        .filter(|row| {
            row.extra_metadata
                .pointer("/voiceNote/noteId")
                .and_then(serde_json::Value::as_str)
                == Some(note_id)
        })
        .collect::<Vec<_>>();
    let [row] = matches.as_slice() else {
        return Err(StartChatError::Other(format!(
            "voice note {note_id} must identify exactly one user row in thread {thread_id}"
        )));
    };
    let mut extra_metadata = row.extra_metadata.clone();
    let Some(metadata) = extra_metadata.as_object_mut() else {
        return Err(StartChatError::Other(
            "voice note row metadata is invalid".into(),
        ));
    };
    if let Some(existing) = metadata
        .get("requestId")
        .and_then(serde_json::Value::as_str)
    {
        if existing == request_id {
            return Ok(());
        }
        return Err(StartChatError::Other(format!(
            "voice note {note_id} is already bound to request {existing}"
        )));
    }
    metadata.insert("requestId".into(), serde_json::json!(request_id));
    blocking::update_message(
        ctx.clone(),
        workspace_dir,
        thread_id.to_string(),
        row.id.clone(),
        ConversationMessagePatch {
            extra_metadata: Some(extra_metadata),
            ..Default::default()
        },
    )
    .await
    .map_err(StartChatError::Other)?;
    Ok(())
}

#[cfg(test)]
#[path = "start_chat_tests.rs"]
mod tests;
