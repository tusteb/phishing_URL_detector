import pytest
import pandas as pd
import numpy as np
from model.transformers import normalize_url, shannon_entropy, parse_url, URLFeatureExtractor, DomainTokenizer

# Тесты для Normalize_url
def test_normalize_url_adds_scheme():
    '''
    Добавляем схему к URL-адресу, если она отсутствует
    '''
    assert normalize_url("example.com") == "http://example.com"

def test_normalize_url_empty_and_nonstring():
    '''
    Проверяем, что функция возвращает пустую строку, когда входными данными 
    являются пустая строка или None
    '''
    assert normalize_url("") == ""
    assert normalize_url(None) == ""

def test_normalize_url_trims_and_cleans():
    '''
    Удаляем пробелы и неразрывные пробелы из URL-адреса и добавляем схему
    '''
    dirty = "  example.com\u200B "
    assert normalize_url(dirty) == "http://example.com"


# Тест для Shannon_entropy
def test_shannon_entropy_simple_cases():
    '''
    Проверяем, что функция правильно рассчитывает энтропию для пустого URL-адреса (0.0) 
    и для URL-адреса, состоящего из повторяющихся символов
    '''
    assert shannon_entropy("") == 0.0
    assert pytest.approx(shannon_entropy("aaaa"), 0.001) == 0.0
    # "ab" -> 2 символа, равные вероятности => энтропия = 1
    assert pytest.approx(shannon_entropy("ab"), 0.001) == 1.0


# Тесты для Parse_url
def test_parse_url_http_and_https():
    '''
    Проверяем, что функция правильно парсит URL-адреса с различными способами доступа
    '''
    parsed = parse_url("example.com")
    assert parsed.scheme == "http"
    assert parsed.netloc == "example.com"

    parsed2 = parse_url("https://test.com/path")
    assert parsed2.scheme == "https"
    assert parsed2.netloc == "test.com"

def test_parse_url_ip_and_ipv6():
    '''
    Проверяем, что функция правильно парсит URL-адреса, содержащие IP-адреса и IPv6-адреса
    '''
    parsed = parse_url("192.168.0.1")
    assert parsed.netloc == "192.168.0.1"

    parsed6 = parse_url("2001:db8::1")
    assert "[" in parsed6.netloc  # IPv6 в квадратных скобках


# Тесты для URLFeatureExtractor
def test_url_feature_extractor_basic():
    '''
    Проверяем, что экстрактор признаков правильно вычленяет признаки из URL-адреса 
    и возвращает их в виде DataFrame
    '''
    extractor = URLFeatureExtractor()
    df = extractor.transform(pd.DataFrame({"url": ["http://example.com/login?x=1&y=2"]}))
    # Проверяем, что все колонки на месте
    assert list(df.columns) == extractor.feature_names_
    # Проверяем отдельные признаки
    row = df.iloc[0].to_dict()
    assert row["has_https"] == 0
    assert row["has_at_symbol"] == 0
    assert row["num_query_params"] == 2
    assert row["has_suspicious_keyword"] == 1  # "login" в пути

def test_url_feature_extractor_ip_address():
    '''
    Проверяем, что функция правильно выявляет URL-ы, содержащие IP-адреса
    '''
    extractor = URLFeatureExtractor()
    df = extractor.transform(pd.DataFrame({"url": ["http://192.168.0.1"]}))
    row = df.iloc[0].to_dict()
    assert row["has_ip"] == 1


# Тесты для DomainTokenizer
def test_domain_tokenizer_basic():
    '''
    Проверяем, что функция правильно токенизирует доменное имя из URL-адреса
    '''
    tokenizer = DomainTokenizer()
    tokens = tokenizer.transform(pd.DataFrame({"url": ["http://sub.example.com"]}))
    assert isinstance(tokens, np.ndarray)
    assert "sub example com" in tokens[0]

def test_domain_tokenizer_multiple_urls():
    '''
    Проверяем, что функция работает правильно с массивом URL-адресов
    '''
    tokenizer = DomainTokenizer()
    urls = ["http://a.com", "http://b-site.org"]
    tokens = tokenizer.transform(pd.DataFrame({"url": urls}))
    assert len(tokens) == 2
    assert "b site org" in tokens[1]