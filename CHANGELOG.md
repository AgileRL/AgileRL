# Changelog

## Unreleased

- The Arena CLI discovers enterprise ``arena on-prem`` commands from ``GET /api/cli/v1/capabilities`` (manifest schema v1). Use an AgileRL release and Arena platform version that implement this endpoint together.
- The ``arena on-prem`` command is hidden unless capabilities grant access: ``enterprise: true`` or ``features.onPremCli: true`` (aligned with when the server emits the manifest).
- On-prem subcommands refetch capabilities when used so enterprise upgrades apply without restarting the CLI.
- On-prem training **submit** remains cloud-only until the platform exposes compatible ``/resources`` entries; capabilities continue to surface ``onPremTrainingSubmit: false`` until then.
