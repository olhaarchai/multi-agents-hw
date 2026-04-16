from typing import Any
import fastmcp
from langchain_core.tools import StructuredTool
from pydantic import create_model, Field


async def mcp_tools_to_langchain(client: fastmcp.Client) -> list[StructuredTool]:
    """Convert all tools on a connected FastMCP Client to LangChain StructuredTools."""
    tools_list = await client.list_tools()
    lc_tools: list[StructuredTool] = []
    for mcp_tool in tools_list:
        tool_name = mcp_tool.name
        tool_desc = mcp_tool.description or tool_name
        input_schema = mcp_tool.inputSchema or {}

        properties = input_schema.get("properties", {})
        required_fields = set(input_schema.get("required", []))
        fields: dict[str, Any] = {}
        for prop_name, prop_schema in properties.items():
            type_str = prop_schema.get("type", "string")
            py_type: type = str
            if type_str == "integer":
                py_type = int
            elif type_str == "number":
                py_type = float
            elif type_str == "boolean":
                py_type = bool
            desc = prop_schema.get("description", "")
            if prop_name in required_fields:
                fields[prop_name] = (py_type, Field(description=desc))
            else:
                fields[prop_name] = (py_type | None, Field(default=None, description=desc))

        ArgsModel = create_model(f"{tool_name}Args", **fields) if fields else None

        async def _tool_fn(_client=client, _name=tool_name, **kwargs) -> str:
            clean = {k: v for k, v in kwargs.items() if v is not None}
            result = await _client.call_tool(_name, clean)
            if result.content:
                return result.content[0].text
            return str(result.data) if result.data is not None else ""

        lc_tools.append(
            StructuredTool.from_function(
                coroutine=_tool_fn,
                name=tool_name,
                description=tool_desc,
                args_schema=ArgsModel,
            )
        )
    return lc_tools
