"""``python -m tts_hub [--host H] [--port P] [--model M] ...``"""
from __future__ import annotations

import argparse
import logging
import os


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS Hub: dubbing service scheduling Breeze TTS 2 and Qwen3-TTS")
    parser.add_argument("--host", help="监听地址（默认 127.0.0.1；对外开放用 0.0.0.0 并设置 --api-key）")
    parser.add_argument("--port", type=int)
    parser.add_argument("--engine", choices=("breeze", "qwen3"), help="默认引擎（请求未指定时使用，启动时预加载）")
    parser.add_argument("--model", help="Breeze 权重：HuggingFace repo id 或本地目录")
    parser.add_argument("--device", help="auto | cuda | cuda:N | cpu")
    parser.add_argument("--data-dir")
    parser.add_argument("--max-batch", type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--idle-unload-minutes", type=float, help="无请求多久后卸载权重（0 = 不卸载，默认 10）")
    parser.add_argument("--idle-shutdown-minutes", type=float, help="卸载后再过多久关闭服务（0 = 不关闭，默认 30）")
    args = parser.parse_args()
    for flag, env in (("host", "TTS_HOST"), ("port", "TTS_PORT"), ("model", "BREEZE_MODEL"),
                      ("engine", "TTS_ENGINE"),
                      ("device", "TTS_DEVICE"), ("data_dir", "TTS_DATA_DIR"),
                      ("max_batch", "TTS_MAX_BATCH"), ("api_key", "TTS_API_KEY"),
                      ("idle_unload_minutes", "TTS_IDLE_UNLOAD_MINUTES"),
                      ("idle_shutdown_minutes", "TTS_IDLE_SHUTDOWN_MINUTES")):
        value = getattr(args, flag)
        if value is not None:
            os.environ[env] = str(value)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    import uvicorn

    from .api import create_app
    from .config import Settings

    settings = Settings.from_env()
    log = logging.getLogger("tts_hub")
    if settings.host not in ("127.0.0.1", "localhost", "::1") and not settings.api_key:
        log.warning("服务监听在 %s 且未设置 API key，任何能访问该端口的人都能调用", settings.host)
    log.info("空闲策略：%s无请求卸载权重，%s后关闭服务",
             f"{settings.idle_unload_s / 60:g} 分钟" if settings.idle_unload_s else "（不卸载）",
             f"再 {settings.idle_shutdown_s / 60:g} 分钟" if settings.idle_shutdown_s else "（不关闭）")
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host=settings.host, port=settings.port, log_level="info"))
    app.state.manager.on_shutdown = lambda: setattr(server, "should_exit", True)
    server.run()


if __name__ == "__main__":
    main()
