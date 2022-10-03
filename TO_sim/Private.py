from knockknock import slack_sender
webhook_url = ("https://hooks.slack.com/services/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")
channel="#python-notification"

MY_slack_sender=slack_sender(webhook_url=webhook_url, channel=channel)