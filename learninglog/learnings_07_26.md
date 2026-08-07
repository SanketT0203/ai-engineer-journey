Exponential backoff
when a call fails do not send another request immediately, instead wait for some time and for each subsequent request wait longer than the lastone. noteworthy funciton `wait_random_ exponential(multiplier=  1,max=30)` is tenacitys builtin function with jitter or randomization. coz if 10 of our requests are hitting the server at same time and get rate limited then same collission is recreated

Timeouts 
stops us from waiting for a reply from a request forever .

429/529 handling
429 is for rate limit and 529 is for api overload.

fallback models

cost caps before api calls(cicuit breaker)

