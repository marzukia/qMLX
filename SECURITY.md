# Security policy

## Supported versions

qMLX is alpha software. Only the current `main` branch is supported. There
are no maintained release lines.

## Reporting a vulnerability

Please do not open a public issue for security problems.

Report vulnerabilities privately via GitHub's private vulnerability
reporting on this repository:
https://github.com/marzukia/qMLX/security/advisories/new

Include what you found, how to reproduce it, and what you think the impact
is. You should get an initial response within a week. This is a solo-
maintained project, so please be patient with fix timelines.

## Scope notes

qMLX inherits Rapid-MLX's server surface. It is intended to run on trusted
networks; the server has no authentication of its own. Binding it to a
public interface is not a supported configuration, and reports that amount
to "the server is reachable" will be closed. Issues in the disk KV
checkpoint path (for example path traversal or unsafe deserialisation) are
very much in scope.

For vulnerabilities in code inherited unchanged from upstream, consider
also reporting to [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) so
the fix lands for everyone.
