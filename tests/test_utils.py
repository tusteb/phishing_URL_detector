import pytest
import types
from ui.utils import load_tlds, is_valid_url_or_ip, is_plausible_url

# Тесты для Load_tlds
def test_load_tlds_success(monkeypatch):
    '''
    Используем monkeypatch, имитируя successful HTTP-запрос на страницу с TLDs, 
    в которой содержатся COM, ORG и RU
    Проверяем, что полученный список TLDs содержит com, org и ru
    '''
    class DummyResp:
        status_code = 200
        text = "# Comment\nCOM\nORG\nRU\n"
        def raise_for_status(self): return None
    monkeypatch.setattr("ui.utils.requests.get", lambda *a, **kw: DummyResp())
    tlds = load_tlds()
    assert "com" in tlds
    assert "org" in tlds
    assert "ru" in tlds

def test_load_tlds_failure(monkeypatch):
    '''
    Используем monkeypatch, имитируя HTTP-запрос, вызывающий исключение
    Проверяем, что функция load_tlds использует механизм fallback и возвращает список 
    TLDs, который должен содержать хотя бы com и ru
    '''
    def bad_get(*a, **kw): raise Exception("fail")
    monkeypatch.setattr("ui.utils.requests.get", bad_get)
    tlds = load_tlds()
    # fallback содержит хотя бы com и ru
    assert "com" in tlds
    assert "ru" in tlds


# Тесты для Is_valid_url_or_ip
@pytest.mark.parametrize("value", ["http://example.com", # валидный URL
                                   "https://example.org", # https
                                   "example.net", # без схемы
                                   "192.168.0.1",  # IPv4
                                   "[2001:db8::1]"]) # IPv6
def test_is_valid_url_or_ip_true(value):
    '''
    Проверка валидных URL/IP
    '''
    assert is_valid_url_or_ip(value) is True

@pytest.mark.parametrize("value", ["", None, "   ", # пустые строки
                                   "not a url", # пробел в netloc
                                   "http://пример.ру", # кириллица в домене
                                   "http://exa mple.com", # пробел внутри домена
                                   "http://exa$mple.com", # недопустимый символ $
                                   "ftp://example.com", # неподдерживаемая схема
                                   "http://"]) # пустой netloc
def test_is_valid_url_or_ip_false(value):
    '''
    Проверка невалидных URL/IP
    '''
    assert is_valid_url_or_ip(value) is False


# Тесты для Is_plausible_url
def test_is_plausible_url_valid(monkeypatch):
    '''
    Проверка валидных URL c подменой VALID_TLDS
    '''
    monkeypatch.setattr("ui.utils.VALID_TLDS", {"com", "org"})
    assert is_plausible_url("http://example.com") is True
    assert is_plausible_url("http://test.org") is True

def test_is_plausible_url_invalid_tld(monkeypatch):
    '''
    Проверка невалидных TLD c подменой VALID_TLDS
    '''
    monkeypatch.setattr("ui.utils.VALID_TLDS", {"com"})
    assert is_plausible_url("http://example.xyz") is False

def test_is_plausible_url_no_latin(monkeypatch):
    '''
    Проверка невалидного URL на кириллице без латиницы c подменой VALID_TLDS
    '''
    monkeypatch.setattr("ui.utils.VALID_TLDS", {"ru"})
    assert is_plausible_url("http://пример.ру") is False

def test_is_plausible_url_bad_format():
    '''
    Проверка невалидного URL с неправильным форматом c подменой VALID_TLDS
    '''
    assert is_plausible_url("not a url") is False
    assert is_plausible_url("http://") is False