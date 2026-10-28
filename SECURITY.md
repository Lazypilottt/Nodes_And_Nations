# Security Policy

## Supported versions

Only the latest commit on `main` receives fixes.

## Reporting a vulnerability

Please **do not** open a public issue for security problems.

Report vulnerabilities privately through GitHub's
[private vulnerability reporting](https://github.com/Lazypilottt/Nodes_And_Nations/security/advisories/new)
("Security" tab → "Report a vulnerability"). Include:

- a description of the issue and its impact,
- steps to reproduce or a proof of concept,
- the affected component (pipeline, dashboard API, prediction API).

You should receive an acknowledgement within 7 days. Once a fix is available we
will credit you in the release notes unless you prefer to stay anonymous.

## Scope notes

The dashboard and prediction API are research tools intended to run locally.
They have no authentication and should not be exposed to the public internet
without putting them behind your own access controls.
