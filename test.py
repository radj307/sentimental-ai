import sentimental

analyzer = sentimental.create_sentiment_analyzer(
    'openai',
    api_key=''
)


def test_print_sentiment(text):
    polarity = analyzer.analyze_sentiment(text)
    print(f'[{polarity}]:\t{text}')


test_print_sentiment('You\'re awesome!')
test_print_sentiment('You\'re okay.')
test_print_sentiment('The sun produces light')
test_print_sentiment('I think you\'re wrong.')
test_print_sentiment('You kinda suck')
test_print_sentiment('You suck!')
test_print_sentiment('Fuck you bitch')
