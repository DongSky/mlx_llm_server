import unittest
from fastapi.testclient import TestClient
from main import app, GenerateRequest, GenerateResponse

class TestGenerateAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_generate_endpoint(self):
        response = self.client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": "你好,请介绍一下你自己。",
                "system": "You are a helpful AI assistant.",
                "context": [],
                "stream": False
            },
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIsInstance(data, dict)
        self.assertIn("response", data)
        self.assertIsInstance(data["response"], str)

    def test_multi_turn_conversation(self):
        response1 = self.client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": "你最喜欢的颜色是什么？",
                "system": "You are a helpful AI assistant.",
                "context": [],
                "stream": False
            },
        )
        self.assertEqual(response1.status_code, 200)

        response2 = self.client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": "为什么你喜欢这个颜色？",
                "system": "You are a helpful AI assistant.",
                "context": [],
                "stream": False
            },
        )
        self.assertEqual(response2.status_code, 200)
        self.assertNotEqual(response1.json()["response"], response2.json()["response"])

    def test_empty_prompt(self):
        response = self.client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": "",
                "system": "You are a helpful AI assistant.",
                "context": [],
                "stream": False
            },
        )
        self.assertEqual(response.status_code, 200)

    def test_long_prompt(self):
        long_prompt = "这是一个非常长的消息" * 100
        response = self.client.post(
            "/api/generate",
            json={
                "model": "Qwen/Qwen2.5-3B-Instruct",
                "prompt": long_prompt,
                "system": "You are a helpful AI assistant.",
                "context": [],
                "stream": False
            },
        )
        self.assertEqual(response.status_code, 200)

    # def test_invalid_model(self):
    #     response = self.client.post(
    #         "/api/generate",
    #         json={
    #             "model": "InvalidModel",
    #             "prompt": "Hello",
    #             "system": "You are a helpful AI assistant.",
    #             "context": [],
    #             "stream": False
    #         },
    #     )
    #     self.assertEqual(response.status_code, 400)

    def test_invalid_request(self):
        response = self.client.post("/api/generate", json={"invalid_key": "invalid_value"})
        self.assertEqual(response.status_code, 422)

if __name__ == "__main__":
    unittest.main()
