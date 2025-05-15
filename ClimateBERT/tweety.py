import tweepy

BEARER_TOKEN = 'YOUR_BEARER_TOKEN_HERE'

client = tweepy.Client(bearer_token=BEARER_TOKEN, wait_on_rate_limit=True)

tweet_id = 'ENTER_TWEET_ID_HERE'

author_id = 'ENTER_AUTHOR_ID_HERE'

def get_replies(tweet_id, author_id, max_results=100):
    query = f'in_reply_to_tweet_id:{tweet_id}'

    replies = []

    paginator = tweepy.Paginator(
        client.search_recent_tweets,
        query=query,
        tweet_fields=['author_id', 'created_at', 'conversation_id', 'in_reply_to_user_id', 'text'],
        max_results= max_results
    )

    for tweet_page in paginator:
        if tweet_page.data is None:
            continue
        for tweet in tweet_page.data:
            replies.append(tweet)
    return replies

if __name__ == '__main__':
    replies = get_replies(tweet_id, author_id)
    print(f"Found {len(replies)} replies to tweet {tweet_id}:\n")
    for reply in replies:
        print(f"- @{reply.author_id} at {reply.created_at}: {reply.text}\n")
