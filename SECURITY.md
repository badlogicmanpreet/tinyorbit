# Security Policy

## The trust model, stated plainly

tinyorbit is an agent that **edits files and runs shell commands on the machine it is running on**,
with arguments chosen by a language model in response to text it was given. That is the entire point
of the tool, and it is also its main risk. Treat it the way you would treat running a script someone
sent you.

Two boundaries exist and are worth understanding before you rely on them.

**The permission chain** resolves every tool call in a fixed order: bypass, then explicit allow
rules, then a read-only classification, then the current mode, then asking the user. Under the
default mode, read-only tools run freely while writes and shell commands prompt. Bash is classified
by splitting the command on pipes and `&&` and treating it as read-only only if every segment starts
with a known reader and carries no redirect or `sudo`. This is a useful guard, not a sandbox, and it
is not a substitute for one.

**The trust boundary** in bootstrap gates whether project files are read at all. Above it, only
`argv` has been touched. An untrusted directory gets no memory files and no auto-allowed writes.

### `--permission-mode bypassPermissions`

This flag disables the permission chain entirely. It exists for disposable containers, which is how
the benchmark harness uses it. **Do not use it on a machine you care about, or on a repository whose
contents you have not read.** Prompt injection is a live concern: an issue description, a README, or
a source comment the agent reads can contain instructions, and under bypass mode there is nothing
between those instructions and your shell.

## Reporting a vulnerability

Please report privately rather than opening a public issue. Use GitHub's private vulnerability
reporting on this repository (Security tab, then "Report a vulnerability"), which notifies the
maintainer without disclosing details publicly.

Useful things to include: what you did, what happened, what you expected, and whether it requires a
particular permission mode. A minimal reproduction against a scratch directory is ideal.

Expect an initial response within a couple of weeks. This is a small project maintained in spare
time; there is no formal SLA and no bounty.

## Scope

In scope: anything that lets a tool call escape the permission chain it should have been subject to,
anything that leaks credentials into logs, transcripts, traces or the container image, and anything
that causes writes outside the working directory without a prompt.

Out of scope: the model choosing to do something unhelpful or wrong while operating within the
permissions it was granted. That is a capability question, not a vulnerability.

## Credentials

tinyorbit reads its API key from the environment or from the Anthropic CLI's stored profile. It never
writes the key to disk. The benchmark harness redacts it from its own logs and excludes local secret
files from the copy it sends into a container. If you find a path where a key reaches a log, a
transcript or a trace file, that is in scope above.
