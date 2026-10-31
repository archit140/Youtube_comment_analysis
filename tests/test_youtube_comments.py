import unittest

import pandas as pd

from src.data.youtube_comments import (
    YouTubeCommentsError,
    extract_video_id,
    fetch_youtube_comments,
)


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params, timeout))
        return next(self.responses)


class YouTubeCommentsTests(unittest.TestCase):
    def test_extracts_supported_youtube_urls(self):
        video_id = "dQw4w9WgXcQ"
        self.assertEqual(extract_video_id(video_id), video_id)
        self.assertEqual(
            extract_video_id(f"https://www.youtube.com/watch?v={video_id}"), video_id
        )
        self.assertEqual(extract_video_id(f"https://youtu.be/{video_id}"), video_id)
        self.assertEqual(
            extract_video_id(f"https://www.youtube.com/shorts/{video_id}"), video_id
        )

    def test_rejects_invalid_url(self):
        with self.assertRaises(ValueError):
            extract_video_id("https://example.com/video")

    def test_fetches_and_paginates_comments(self):
        first = {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "textOriginal": "Great video",
                                "authorDisplayName": "A",
                                "likeCount": 4,
                                "publishedAt": "2026-01-01T00:00:00Z",
                            }
                        }
                    }
                }
            ],
            "nextPageToken": "next",
        }
        second = {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {
                            "snippet": {
                                "textDisplay": "Helpful explanation",
                                "authorDisplayName": "B",
                                "likeCount": 1,
                                "publishedAt": "2026-01-02T00:00:00Z",
                            }
                        }
                    }
                }
            ]
        }
        session = FakeSession([FakeResponse(first), FakeResponse(second)])
        frame = fetch_youtube_comments(
            "https://youtu.be/dQw4w9WgXcQ", "test-key", max_comments=2, session=session
        )
        self.assertIsInstance(frame, pd.DataFrame)
        self.assertEqual(frame["comment"].tolist(), ["Great video", "Helpful explanation"])
        self.assertEqual(len(session.calls), 2)
        self.assertEqual(session.calls[1][1]["pageToken"], "next")

    def test_requires_api_key(self):
        with self.assertRaises(YouTubeCommentsError):
            fetch_youtube_comments("dQw4w9WgXcQ", "")


if __name__ == "__main__":
    unittest.main()
