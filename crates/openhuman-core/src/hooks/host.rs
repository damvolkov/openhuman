//! The process-global hook engine and the OpenHuman seams it is built with.
//!
//! The engine, `hooks.json` contract and process runner live in
//! `tinyagents_hooks::command`. This module supplies what is specific
//! to this host: the product name that decides discovery paths and the
//! `OPENHUMAN_*` hook environment, the user's home directory, the platform
//! shell, and the model behind `prompt` hooks.

use crate::core::runtime::DispatchCtx;
use std::sync::{Arc, LazyLock};

use async_trait::async_trait;
use tinyagents_hooks::command::{config, HookConfig, HookEngine, HookEnvironment, PromptEvaluator};

/// Product name handed to the engine: `ProgramData\OpenHuman`,
/// `/Library/Application Support/OpenHuman`, `/etc/openhuman`, `.openhuman`,
/// and the `OPENHUMAN_*` hook variables all derive from it.
pub const PRODUCT_NAME: &str = "OpenHuman";

struct HostPromptEvaluator;

#[async_trait]
impl PromptEvaluator for HostPromptEvaluator {
    async fn evaluate(&self, instruction: &str, model: Option<&str>) -> Result<String, String> {
        super::prompt_eval::evaluate(&DispatchCtx::ambient(), instruction, model).await
    }
}

fn environment() -> HookEnvironment {
    HookEnvironment::new(PRODUCT_NAME, dirs::home_dir())
        .with_shell(super::shell::command)
        .with_prompt_evaluator(Arc::new(HostPromptEvaluator))
}

static ENGINE: LazyLock<Arc<HookEngine>> =
    LazyLock::new(|| Arc::new(HookEngine::new(environment())));

/// The process-global engine.
pub fn engine() -> &'static Arc<HookEngine> {
    &ENGINE
}

/// Re-read every `hooks.json` layer and install the result.
///
/// The engine's own [`HookEngine::reload`] reuses the environment it was built
/// with, which froze the home directory at first use. Discovery resolves the
/// user layer against the home directory **now**, as it did before the engine
/// moved upstream, so a reload after `HOME` changes reads the right file.
pub async fn reload(
    ctx: &DispatchCtx,
    project_dir: Option<std::path::PathBuf>,
    workspace_dir: Option<std::path::PathBuf>,
) -> Arc<HookConfig> {
    // Hook commands run under the shell TinyBox selects; ask it here, where the
    // caller is async, so the engine's synchronous shell factory can use it.
    super::shell::resolve(ctx).await;
    let current = environment();
    let loaded = tokio::task::spawn_blocking(move || {
        config::load(&current, project_dir.as_deref(), workspace_dir.as_deref())
    })
    .await
    .unwrap_or_default();
    for warning in &loaded.warnings {
        log::warn!("[hooks] {warning}");
    }
    log::info!(
        "[hooks] active configuration: {} hook(s) across {} file(s)",
        loaded.len(),
        loaded.sources.len()
    );
    ENGINE.install(loaded).await;
    ENGINE.snapshot().await
}

/// Prompt that makes the test `beforeSubmitPrompt` hooks add context.
#[cfg(test)]
pub(crate) const TEST_HOOK_CONTEXT: &str = "ZZ-HOOK-CONTEXT";
/// Prompt that makes the test `beforeSubmitPrompt` hooks block.
#[cfg(test)]
pub(crate) const TEST_HOOK_BLOCK: &str = "ZZ-HOOK-BLOCK";

/// Install, once per process and behind a POSIX shell, `beforeSubmitPrompt`
/// hooks that act only on prompts carrying [`TEST_HOOK_CONTEXT`] or
/// [`TEST_HOOK_BLOCK`], so no other test's prompt meets them. One installer for
/// every test module: the engine is process-wide and `install` replaces it.
#[cfg(test)]
pub(crate) async fn install_prompt_hooks_for_test() {
    static INSTALLED: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();
    INSTALLED
        .get_or_init(|| async {
            let hook = |command: &str, matcher: &str| {
                serde_json::from_value::<tinyagents_hooks::command::HookDefinition>(
                    serde_json::json!({ "command": command, "matcher": matcher }),
                )
                .expect("test hook definition")
            };
            let mut config = HookConfig::default();
            config.by_event.insert(
                tinyagents_hooks::command::HookEvent::BeforeSubmitPrompt,
                vec![
                    hook(
                        r#"cat >/dev/null; echo '{"additional_context": "EXTRACTED-BY-HOOK"}'"#,
                        &format!("^{TEST_HOOK_CONTEXT}"),
                    ),
                    hook(
                        r#"cat >/dev/null; echo '{"continue": false, "user_message": "hook says no"}'"#,
                        &format!("^{TEST_HOOK_BLOCK}"),
                    ),
                ],
            );
            super::shell::install_posix_for_test();
            ENGINE.install(config).await;
        })
        .await;
}
