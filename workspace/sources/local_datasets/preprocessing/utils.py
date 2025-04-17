import pandas as pd
import nltk
import string
import re
import numpy as np
from nltk import ngrams
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from czech_stemmer import cz_stem
from stop_words import get_stop_words
import simplemma


'''
This is a list of words that shall never be lowercased beacause of their ambiguous meaning.
'''
global untouchable_words
untouchable_words = ['US']


def remove_punctation(news_articles: pd.Series) -> pd.Series:
    '''
    Removes punctation from all news articles in the given series (one-dimensional ndarray).

    Parameters
    ----------
    news_articles: pandas.Series
            news_articles: Series (one-dimensional ndarray) of news articles.

    Returns
    -------
    pandas.Series
        Series of articles stripped of punctation.
    '''
    all_punctuation = string.punctuation
    all_punctuation += 'ʺ-„–‘’“”—✔️©'
    return news_articles.str.translate(str.maketrans(all_punctuation, ' ' * len(all_punctuation)))


def remove_urls(news_articles: pd.Series) -> pd.Series:
    '''
    Removes urls from all news articles in the given series (one-dimensional ndarray).

    Parameters
    ----------
    pandas.Series
            Series (one-dimensional ndarray) of news articles.

    Returns
    -------
    pandas.Series
        Series of articles stripped of urls.
    '''
    return news_articles.replace(r'http\S+', '', regex=True).replace(r'www\S+', ' ', regex=True)


def remove_numbers(news_articles: pd.Series) -> pd.Series:
    '''
    Removes numerical characters from all news articles in the given series (one-dimensional ndarray).

    Parameters
    ----------
    news_articles: pandas.Series
            Series (one-dimensional ndarray) of news articles.

    Returns
    -------
    pandas.Series
        Series of articles stripped of numbers.
    '''
    return news_articles.str.replace('\d+', '')


def remove_special_characters(news_articles: pd.Series) -> pd.Series:
    '''
    Removes all unnecessary characters (numerals, punctation, urls) from all news articles in the given series (one-dimensional ndarray).

    Parameters
    ----------
    news_articles: pandas.Series
            Series (one-dimensional ndarray) of news articles.

    Returns
    -------
    pandas.Series
        Series of articles stripped of all unnecessary characters.
    '''
    return remove_numbers(remove_punctation(remove_urls(news_articles)))


def truncate_articles(news_artricles: pd.Series, length: int) -> pd.Series:
    '''
    Truncates articles to desired length.

    Parameters
    ----------
    news_articles: pandas.Series
            Series (one-dimensional ndarray) of news articles.
    length: int
            Maximal allow number of words in articles.

    Returns
    -------
    pandas.Series
        Series of articles stripped of all unnecessary characters.
    '''

    return news_artricles.apply(lambda row: row[:length])


def create_ngrams(row: str, gram_size: int) -> list:
    '''
    Lambda function which is being applied to concrete news article.

    Parameters
    ----------
    row: str
            News article in form of string.
    gram_size: int
            Maximal size of n-gram allowed in tokenization.

    Returns
    -------
    list
        Article converted into list of token of maximal size.
    '''
    tokens = [t.lower() if t not in untouchable_words else t for t in word_tokenize(row)]
    # result = list(ngrams(tokens, gram_size))
    return [list(x) for idx, x in enumerate(list(ngrams(tokens, gram_size)))]
    # for idx,x in enumerate(result):
    #     result[idx] = list(x)
    # return result


def tokenize(news_articles: pd.Series, gram_size: int) -> pd.Series:
    '''
    Tokenizes all texts into n-grams with maximal size given in parameter using lambda function.

    Parameters
    ----------
    news_articles: pandas.Series
           news_articles: Series (one-dimensional ndarray) of news articles.
    gram_size: int
            Maximal size of n-gram allowed in tokenization.

    Returns
    -------
    pandas.Series
        Series of articles tokenized into n-grams with given size.
    '''
    return news_articles.apply(lambda row: " ".join(row)).apply(lambda row: create_ngrams(row, gram_size))


def remove_stop_words(news_articles: pd.Series, language: str = 'en') -> pd.Series:
    '''
    Removes stop words from pre-tokenized news articles in given language.
    If the entered language is not supported an exception is raised. Finally in removes empty tokens from articles.

    Parameters
    ----------
    news_articles: pandas.Series
            Series (one-dimensional ndarray) of already tokenized news articles.
    language: str
            Language used in articles. Available options are 'en' or 'cs'.

    Rises
    -----
    ValueError
            If the entered language is not supported.

    Returns
    -------
    pandas.Series
        Series of articles stripped of stop words.
    '''

    if language == 'en':
        stop_words = [word for word in nltk.corpus.stopwords.words('english')]
    elif language == 'cs':
        stop_words = get_stop_words('czech')
    else:
        raise ValueError("You have entered the wrong language!")

        # add empty string to set of stpowords
    stop_words.append('')
    stop_words = set(stop_words)

    return news_articles.apply(lambda row: [w.lower() if w not in untouchable_words else w for w in row.split() if
                                            w.lower() not in stop_words])


def stem(news_articles: pd.Series, language: str = 'en', embedding: bool = False) -> pd.Series:
    '''
    Stems the tokens in the articles by specified language. Uses SnowballStemmer of english language and light stemming for the czech language.
    If the entered language is not supported an exception is raised.

    Parameters
    ----------
    news_articles: pandas.Series
            Series (one-dimensional ndarray) of already tokenized news articles.
    language: string
            Language used in articles. Available options are 'en' or 'cs'.
    embedding: bool
            Tells us if result data will go into feature extraction model or word embedding. Default value is False.

    Rises
    -----
    ValueError
            If the provided language in not supported.

    Returns
    -------
    pandas.Series
        Series of stemmed tokenized articles.
    '''

    if language not in ['en', 'cs']:
        raise ValueError("You have entered the wrong language!")

    if embedding:
        if language == 'en':
            snow = SnowballStemmer("english")
            return news_articles.apply(lambda row: [snow.stem(word) for word in row])
        elif language == 'cs':
            return news_articles.apply(lambda row: [cz_stem(word) for word in row])

    if language == 'en':
        snow = SnowballStemmer("english")
        return news_articles.apply(lambda row: [[snow.stem(word) for word in ngram] for ngram in row])
    elif language == 'cs':
        return news_articles.apply(lambda row: [[cz_stem(word) for word in ngram] for ngram in row])


def lemmatize(news_articles: pd.Series, language: str = 'en', embedding: bool = False) -> pd.Series:
    '''
    Lemmatize the tokens in the articles by specified language. Uses simplemma lemmatizator for both languages.
    If the entered language is not supported an exception is raised.

    Parameters
    ----------
    news_articles: pandas.Series
            Series (one-dimensional ndarray) of already tokenized news articles.
    language: string
            Language used in articles. Available options are 'en' or 'cs'.
    embedding: bool
            Tells us if result data will go into feature extraction model or word embedding. Default value is False.

    Rises
    -----
    ValueError
            If the provided language in not supported.

    Returns
    -------
    pandas.Series
        Series of lemmatized tokenized articles.

    '''
    if language not in ['en', 'cs']:
        raise ValueError("You have entered the wrong language!")

    if embedding:
        return news_articles.apply(lambda row: [simplemma.lemmatize(word, lang=language) for word in row])

    return news_articles.apply(
        lambda row: [[simplemma.lemmatize(word, lang=language) for word in ngram] for ngram in row])
