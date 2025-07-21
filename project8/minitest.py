from ManualStrategy import ManualStrategy
from marketsimcode import compute_portvals
import datetime as dt

manual = ManualStrategy()
trades = manual.testPolicy(sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31))
portvals = compute_portvals(trades)

print(portvals.head())