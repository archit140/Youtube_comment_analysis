"""Optional YouTube comment retrieval for the local sentiment dashboard.

The model remains local. This module only retrieves top-level comments when a
user explicitly supplies a YouTube URL and an API key.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests


YOUTUBE_COMMENT_THREADS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
VIDEO_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")


class YouTubeCommentsError(RuntimeError):
    """Raised when a YouTube URL or comment request cannot be processed."""


def extract_video_id(value: str) -> str:
    """Extract and validate a YouTube video ID from a URL or bare ID."""
    candidate = str(value or "").strip()
    if VIDEO_ID_PATTERN.fullmatch(candidate):
        return candidate

    parsed = urlparse(candidate)
    host = parsed.netloc.lower().split(":", 1)[0]
    video_id = ""
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host.endswith("youtube.com"):
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if not video_id:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 2 and parts[0] in {"shorts", "embed", "live"}:
                video_id = parts[1]

    if not VIDEO_ID_PATTERN.fullmatch(video_id):
        raise ValueError("Enter a valid YouTube watch URL, short URL, or video ID.")
    return video_id


def fetch_youtube_comments(
    video_url: str,
    api_key: str,
    *,
    max_comments: int = 500,
    timeout: int = 20,
    session=None,
) -> pd.DataFrame:
    """Fetch up to ``max_comments`` top-level comments into a DataFrame.

    ``session`` is injectable so tests can use a fake response without making
    network requests.
    """
    if not api_key or not str(api_key).strip():
        raise YouTubeCommentsError(
            "YOUTUBE_API_KEY is not configured. Add it to your local .env file."
        )
    if max_comments < 1:
        raise ValueError("max_comments must be at least 1.")

    video_id = extract_video_id(video_url)
    http = session or requests.Session()
    comments: list[dict] = []
    page_token = None

    while len(comments) < max_comments:
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": min(100, max_comments - len(comments)),
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token
        try:
            response = http.get(YOUTUBE_COMMENT_THREADS_URL, params=params, timeout=timeout)
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            raise YouTubeCommentsError("Could not retrieve comments from YouTube.") from exc
        except ValueError as exc:
            raise YouTubeCommentsError("YouTube returned an invalid response.") from exc

        for item in payload.get("items", []):
            try:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
            except (KeyError, TypeError):
                continue
            comments.append(
                {
                    "comment": snippet.get("textOriginal") or snippet.get("textDisplay", ""),
                    "author": snippet.get("authorDisplayName", ""),
                    "like_count": int(snippet.get("likeCount", 0)),
                    "published_at": snippet.get("publishedAt", ""),
                }
            )
            if len(comments) >= max_comments:
                break

        page_token = payload.get("nextPageToken")
        if not page_token or not payload.get("items"):
            break

    frame = pd.DataFrame(comments, columns=["comment", "author", "like_count", "published_at"])
    if frame.empty:
        raise YouTubeCommentsError("No top-level comments were returned for this video.")
    return frame
