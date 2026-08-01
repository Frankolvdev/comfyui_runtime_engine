import asyncio


class FakeQueue:
    pass


class FakePromptServer:
    def __init__(self) -> None:
        self.prompt_queue = FakeQueue()


async def _run_forever() -> None:
    await asyncio.Event().wait()


def start_comfyui(asyncio_loop=None):
    loop = asyncio_loop or asyncio.new_event_loop()
    server = FakePromptServer()

    async def start_all():
        await _run_forever()

    return loop, server, start_all


def cleanup_temp():
    return None
