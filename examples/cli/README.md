# Ralph CLI Example

> **Note**: This package is experimental. APIs may change between versions.

A general-purpose autonomous coding agent for long-running tasks like:

- **Code migrations** (Jest → Vitest, CJS → ESM, etc.)
- **Dependency upgrades** (React 17 → 18, TypeScript 4 → 5, etc.)
- **Large refactoring** (rename across codebase, restructure directories)
- **Feature implementation** (from specifications)
- **Bug fixes** (across multiple files)

## Usage

```bash
# Interactive mode - will interview you to define the task
pnpm start -- /path/to/project

# With an inline prompt
pnpm start -- /path/to/project "Migrate from CommonJS to ESM"

# With a prompt file
pnpm start -- /path/to/project ./my-task.md
```

## Interactive Mode

If no `PROMPT.md` exists and no prompt is provided, the CLI will interview you with **AI-powered suggestions**:

```
? What type of task is this? › 
❯   Create - Create a new project, app, or library from scratch
    Migration - Migrate between frameworks, libraries, or patterns
    Upgrade - Upgrade dependencies or language versions
    ...

? Give your task a short title: › Migration: Jest to Vitest

Analyzing codebase...
  Generating suggestions...

? What needs to be done? ›
❯   Convert all Jest test files to use Vitest syntax and assertions
    Update test configuration from jest.config.js to vitest.config.ts
    Replace Jest mocking utilities with Vitest equivalents
    ✏️  Other (enter custom)

? What context is important? ›
❯   TypeScript project using ES modules, tests located in __tests__ directories
    Using React Testing Library for component tests
    CI pipeline runs tests with coverage requirements
    ✏️  Other (enter custom)

? Where should the agent focus? ›
❯   src/__tests__/, tests/, *.test.ts files
    jest.config.js, package.json test scripts
    Mock files in __mocks__ directories
    ✏️  Other (enter custom)

? How should success be verified? ›  (user selects)
◉   Run tests
◉   Type check (tsc)
◯   Lint
◯   Build

? What does success look like? ›
❯   All 150 existing tests pass with Vitest, no Jest dependencies remain
    Test coverage remains at or above 80%
    CI pipeline passes with new test runner
    ✏️  Other (enter custom)

? Save as PROMPT.md in the target directory? › yes
```

The AI analyzes your codebase and generates contextual suggestions for each question. Select a suggestion or choose "Other" to enter custom text.

## Features

### Context Management
Handles long conversations automatically:
- **Auto-summarization** of older iterations when approaching token limits
- **Large file handling** with line-range reading
- **Change log** tracking decisions and progress

### Efficient Editing
- `editFile` tool for surgical search/replace (more token-efficient than full rewrites)
- `readFile` with `lineStart`/`lineEnd` for reading specific sections of large files

### Tools Available
| Tool | Description |
|------|-------------|
| `listFiles` | Glob-based file listing |
| `readFile` | Read files (with optional line range) |
| `writeFile` | Write/create files |
| `editFile` | Search/replace editing |
| `deleteFile` | Delete files |
| `runCommand` | Execute shell commands |
| `markComplete` | Signal task completion |

## Environment

```bash
export ANTHROPIC_API_KEY=your_key_here
```
