# Changelog

## Unreleased

### On-prem CLI (manifest v2 — breaking)

Requires an Arena platform build that serves ``manifestSchemaVersion: 2`` from ``GET /api/cli/v1/capabilities``. Use a matching AgileRL release.

**Command tree**

- ``providers`` — ``enable``, ``disable``, ``get``, ``update``
- ``classes`` — ``list``, ``create``, ``get``, ``update``, ``delete``
- ``install`` — ``bundle`` (worker archive download), ``bootstrap`` (enable → create class → download bundle)

**Breaking path changes** (no deprecated aliases):

| Old | New |
|-----|-----|
| ``arena on-prem enable`` / ``disable`` | ``arena on-prem providers enable`` / ``disable`` |
| ``arena on-prem provider get`` / ``update`` | ``arena on-prem providers get`` / ``update`` |
| ``arena on-prem classes deployment-setup`` | ``arena on-prem install bundle`` |

**Golden path**

```bash
arena on-prem install bootstrap \
  --name worker-pool --num-nodes 4 \
  --setup-type helm --cpus 8 --gpus 1 --memory "64 GB" \
  -o ./arena-workers.zip
```

- ``arena on-prem`` is shown when capabilities have ``enterprise: true`` **or** ``features.onPremCli: true``.
- On-prem subcommands refetch capabilities when used so entitlement changes apply without restarting the CLI.
- On-prem training **submit** remains cloud-only until the platform exposes compatible ``/resources`` entries; capabilities continue to surface ``onPremTrainingSubmit: false`` until then.
