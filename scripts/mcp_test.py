"""Drive mcp_server.py over stdio the way an agent would (service must be running)."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

ROOT = Path(__file__).resolve().parents[1]


async def main() -> int:
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")],
                                   env={"PYTHONUTF8": "1"})
    out = Path(tempfile.mkdtemp(prefix="tts_mcp_"))
    async with stdio_client(params) as (read, write), ClientSession(read, write) as s:
        await s.initialize()
        tools = [t.name for t in (await s.list_tools()).tools]
        print("tools:", tools)

        async def call(tool, **kw):
            r = await s.call_tool(tool, kw)
            text = r.content[0].text if r.content else ""
            if r.isError:
                raise RuntimeError(f"{tool}: {text}")
            return json.loads(text) if text.startswith(("{", "[")) else text

        voices = await s.call_tool("tts_list_voices", {})
        print("voices:", len(voices.content))
        r = await call("tts_speak", text="通过 MCP 合成的一句话。", voice="zh_young_male",
                       output_path=str(out / "mcp_tts.wav"))
        print("tts:", r["file"], r["duration_s"], Path(r["file"]).is_file())
        r = await call("tts_dub", name="mcp_dub", output_dir=str(out / "dub"), lines=[
            {"voice": "zh_male_narrator", "text": "旁白：故事开始了。"},
            {"voice": "zh_girl", "text": "妈妈，我们去哪儿呀？", "instruction": "好奇地问"},
        ])
        print("dub:", r["file"], r["srt"], [l["file"] for l in r["lines"]])
        assert Path(r["file"]).is_file() and Path(r["srt"]).is_file()
    print("output dir:", out)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
