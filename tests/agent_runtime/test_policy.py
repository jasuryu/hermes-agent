"""Tests for Agent Runtime command and approval policy primitives."""

from __future__ import annotations

from agent_runtime import policy


def test_ops_read_only_commands_are_allowed_without_approval():
    verdict = policy.classify_command("kubectl -n prod get pods")

    assert verdict.allowed_without_approval is True
    assert verdict.requires_approval is False
    assert verdict.category == "read_only"


def test_secret_bearing_read_commands_require_approval():
    for command in (
        "kubectl -n prod get secrets",
        "kubectl describe secret api-token",
        "kubectl logs deploy/api",
        "kubectl config view --raw",
        "kubectl get --raw /api/v1/namespaces/prod/secrets/api-token",
        "terraform output",
        "terraform state pull",
        "terraform state show module.db",
        "docker inspect api-container",
        "docker logs api-container",
    ):
        verdict = policy.classify_command(command)
        assert verdict.allowed_without_approval is False
        assert verdict.requires_approval is True
        assert verdict.category == "sensitive_read"


def test_helm_secret_bearing_release_reads_require_approval():
    for command in (
        "helm get values api -n prod",
        "helm get manifest api -n prod",
        "helm get all api -n prod",
        "helm get hooks api -n prod",
        "helm get notes api -n prod",
        "helm template api ./chart -n prod",
        "helm show values ./chart",
        "helm show all ./chart",
        "kubectl -n prod get pods -o yaml",
        "kubectl -n prod get pods -oyaml",
        "kubectl -n prod get pods -ojsonpath={.items[*].spec.containers[*].env}",
        "kubectl -n prod get pods --template={{.metadata.name}}",
        "kubectl get --raw /api/v1/namespaces/prod/configmaps/app-config",
        "kubectl get --raw=/api/v1/namespaces/prod/configmaps/app-config",
        "kubectl config view --raw=true",
        "kubectl describe deploy/api",
        "kubectl get configmap app-config -o yaml",
        "docker ps",
        "docker top api-container",
    ):
        verdict = policy.classify_command(command)
        assert verdict.allowed_without_approval is False
        assert verdict.requires_approval is True
        assert verdict.category == "sensitive_read"


def test_prod_mutation_commands_require_approval_packet():
    verdict = policy.classify_command("kubectl -n prod rollout restart deploy/api")

    assert verdict.allowed_without_approval is False
    assert verdict.requires_approval is True
    assert verdict.category == "prod_mutation"
    assert "approval packet" in verdict.reason.lower()


def test_destructive_cleanup_requires_approval_even_outside_kubectl():
    verdict = policy.classify_command("docker system prune -af")

    assert verdict.requires_approval is True
    assert verdict.category == "destructive"


def test_approval_scope_matches_exact_command_hash():
    command = "kubectl -n prod rollout restart deploy/api"
    approval = policy.build_approval_packet(
        target="cluster=whale namespace=prod deploy/api",
        commands=[command],
        reason="restart stuck deployment",
        blast_radius="api pods restart",
        rollback="kubectl -n prod rollout undo deploy/api",
        verification=["kubectl -n prod rollout status deploy/api"],
        approved_by="Jasur",
    )

    assert policy.approval_allows_command(
        approval,
        command,
        target="cluster=whale namespace=prod deploy/api",
    ) is True
    assert policy.approval_allows_command(
        approval,
        command,
        target="cluster=whale namespace=staging deploy/api",
    ) is False
    assert policy.approval_allows_command(
        approval,
        "kubectl -n prod delete deploy/api",
        target="cluster=whale namespace=prod deploy/api",
    ) is False

    forged = dict(approval)
    forged["target"] = "cluster=whale namespace=stage deploy/api"
    assert policy.approval_allows_command(
        forged,
        command,
        target="cluster=whale namespace=prod deploy/api",
    ) is False

    forged_hashes = dict(approval)
    forged_hashes["command_hashes"] = approval["command_hashes"] + [policy.command_hash("kubectl -n prod delete deploy/api")]
    assert policy.approval_allows_command(
        forged_hashes,
        "kubectl -n prod delete deploy/api",
        target="cluster=whale namespace=prod deploy/api",
    ) is False


def test_approval_allows_command_requires_exact_command_string(monkeypatch):
    target = "cluster=whale namespace=prod deploy/api"
    approved = "kubectl -n prod rollout restart deploy/api"
    different = "kubectl -n prod delete deploy/api"
    monkeypatch.setattr(policy, "command_hash", lambda _command: "forced-collision")
    approval = policy.build_approval_packet(
        target=target,
        commands=[approved],
        reason="maintenance",
        blast_radius="api deploy",
        rollback="rollout undo",
        verification=["kubectl get deploy/api"],
        approved_by="jasur",
        expires_at=9_999_999_999,
    )

    assert policy.approval_allows_command(approval, approved, target=target) is True
    assert policy.approval_allows_command(approval, different, target=target) is False


def test_approval_scope_hash_is_not_newline_ambiguous():
    first = policy.approval_scope_hash("cluster=one\nnamespace=prod", ["kubectl get pods"])
    second = policy.approval_scope_hash("cluster=one", ["namespace=prod", "kubectl get pods"])

    assert first != second


def test_compound_shell_commands_fail_closed_even_with_readonly_first_segment():
    verdict = policy.classify_command("kubectl get pods && kubectl scale deploy/api --replicas=0")

    assert verdict.allowed_without_approval is False
    assert verdict.requires_approval is True
    assert verdict.category == "compound_shell"


def test_rm_recursive_force_variants_require_approval():
    verdict = policy.classify_command("rm -fr /tmp/something")

    assert verdict.requires_approval is True
    assert verdict.category == "destructive"


def test_newline_separated_shell_commands_fail_closed_before_canonicalization():
    verdict = policy.classify_command("kubectl get pods\nkubectl scale deploy/api --replicas=0")

    assert verdict.requires_approval is True
    assert verdict.category == "compound_shell"


def test_unknown_commands_fail_closed_until_explicitly_allowlisted():
    verdict = policy.classify_command("systemctl restart postgresql")

    assert verdict.allowed_without_approval is False
    assert verdict.requires_approval is True
    assert verdict.category == "unknown_command"


def test_allowlisted_ops_tokens_inside_unknown_executables_fail_closed():
    for command in (
        "python kubectl get pods",
        "make helm template",
        "./wrapper docker ps",
        "bash terraform plan",
    ):
        verdict = policy.classify_command(command)
        assert verdict.allowed_without_approval is False
        assert verdict.requires_approval is True
        assert verdict.category == "unknown_command"


def test_kubectl_auth_and_config_mutations_require_approval():
    assert policy.classify_command("kubectl auth reconcile -f rbac.yaml").requires_approval is True
    assert policy.classify_command("kubectl config set-context prod").requires_approval is True


def test_terraform_state_and_workspace_mutations_require_approval():
    assert policy.classify_command("terraform state mv a b").requires_approval is True
    assert policy.classify_command("terraform state push tfstate.json").requires_approval is True
    assert policy.classify_command("terraform workspace select prod").requires_approval is True


def test_approval_hash_preserves_shell_significant_whitespace():
    approved = "kubectl get pods"
    different = "kubectl get\npods"
    leading = " kubectl get pods"

    approval = policy.build_approval_packet(
        target="cluster=whale namespace=prod",
        commands=[approved],
        reason="read check",
        blast_radius="none",
        rollback="none",
        verification=[approved],
        approved_by="Jasur",
    )

    assert policy.approval_allows_command(approval, approved, target="cluster=whale namespace=prod") is True
    assert policy.approval_allows_command(approval, different, target="cluster=whale namespace=prod") is False
    assert policy.approval_allows_command(approval, leading, target="cluster=whale namespace=prod") is False


def test_shell_redirection_and_background_operators_fail_closed():
    assert policy.classify_command("kubectl get pods > /tmp/pods.txt").requires_approval is True
    assert policy.classify_command("kubectl get pods 2> /tmp/errors.txt").requires_approval is True
    assert policy.classify_command("kubectl get pods &").requires_approval is True
