# Configuration Documentation

## MCP Configuration

### Workspace-level

The workspace uses this MCP config file:

- `.vscode/mcp.json`

Current server configuration:

```json
{
  "servers": {
    "docs-langchain": {
      "url": "https://docs.langchain.com/mcp"
    }
  }
}
```

### User-level status (as of 2026-03-21)

User-level `mcp.servers` has been removed from both:

- WSL VS Code remote user settings
- Windows VS Code user settings

This means MCP is currently configured at workspace scope only.
