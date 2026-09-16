#!/usr/bin/env python3

from api.services.pipeline import CorePipelineService


def main() -> None:
    service = CorePipelineService()
    stats = service.refresh()
    print(stats)


if __name__ == "__main__":
    main()

