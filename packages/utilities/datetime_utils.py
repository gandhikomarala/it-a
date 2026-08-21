"""Datetime and timestamp utility functions."""
from datetime import datetime, date, timedelta, timezone
from typing import Tuple

def utc_now() -> datetime:
    """Return timezone-aware current UTC datetime."""
    return datetime.now(timezone.utc)

def format_iso_datetime(dt: datetime) -> str:
    """Format datetime to standard ISO 8601 UTC string."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()

def parse_iso_datetime(iso_str: str) -> datetime:
    """Parse standard ISO 8601 string into UTC datetime."""
    dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt

def get_date_window_days(days: int = 30) -> Tuple[datetime, datetime]:
    """Return (start_datetime, end_datetime) for a trailing window of N days."""
    end = utc_now()
    start = end - timedelta(days=days)
    return start, end

def calculate_age_years(born_date: date) -> int:
    """Calculate current age in years given a birthdate."""
    today = date.today()
    return today.year - born_date.year - ((today.month, today.day) < (born_date.month, born_date.day))
