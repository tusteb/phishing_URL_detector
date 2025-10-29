import pytest
import numpy as np
from fastapi.testclient import TestClient
import types
import api.api

client = TestClient(api.api.app)

# Фикстура для подмены pipeline
@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    class DummyPipeline:
        def predict_proba(self, X):
            # Всегда возвращаем вероятность 0.9 для класса "фишинг"
            return np.array([[0.1, 0.9]])
    monkeypatch.setattr(api.api, "pipeline", DummyPipeline())
    yield

# Тесты для /predict
def test_predict_valid_url():
    '''
    Отправляем валидный URL и проверяем, что ответ 200 и содержит корректные поля
    '''
    resp = client.get("/predict", params={"url": "http://example.com"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["url"].startswith("http://")
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)

def test_predict_invalid_url():
    '''
    Отправляем строку, которая не является URL, ожидаем ошибку 400 и сообщение 
    о некорректном формате
    '''
    resp = client.get("/predict", params={"url": "not a url"})
    assert resp.status_code == 400
    assert "Некорректный формат URL" in resp.json()["detail"]

def test_predict_empty_url():
    '''
    Отправляем пустую строку, ожидаем ошибку 400 и сообщение, что URL не может быть пустым
    '''
    resp = client.get("/predict", params={"url": ""})
    assert resp.status_code == 400
    assert "URL не может быть пустым" in resp.json()["detail"]

# Тесты для /explain
def test_explain_valid_url():
    '''
    Отправляем валидный URL, проверяем, что получается список объяснений
    '''
    resp = client.get("/explain", params={"url": "http://example.com"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["url"].startswith("http://")
    assert "explanations" in data
    assert isinstance(data["explanations"], list)
    assert any("feature" in h for h in data["explanations"])

def test_explain_invalid_url():
    '''
    Отправляем строку, которая не является URL, ожидаем ошибку 400 и сообщение 
    о некорректном формате
    '''
    resp = client.get("/explain", params={"url": "not a url"})
    assert resp.status_code == 400
    assert "Некорректный формат URL" in resp.json()["detail"]

def test_explain_empty_url():
    '''
    Отправляем пустую строку, ожидаем ошибку 400 и сообщение, что URL не может быть пустым
    '''
    resp = client.get("/explain", params={"url": ""})
    assert resp.status_code == 400
    assert "URL не может быть пустым" in resp.json()["detail"]