import datetime


def utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)