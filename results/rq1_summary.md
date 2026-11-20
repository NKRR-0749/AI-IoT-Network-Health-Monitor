# RQ1 summary: how injected impairments show up (16 mixed runs, EMULATED userspace proxy)

Mean over all phases of each kind. Outage phases count as 100% injected loss.

| phase        |   qos |   phases |   udp_loss_pct |   mqtt_msg_loss_pct |   rtt_median_ms |   disconnects_per_phase |
|:-------------|------:|---------:|---------------:|--------------------:|----------------:|------------------------:|
| delay        |     0 |        6 |           0    |                0    |          598.77 |                    0    |
| delay        |     1 |        6 |           0    |                0    |          609.25 |                    0    |
| outage_drop  |     0 |        5 |         100    |              100    |          nan    |                   10    |
| outage_drop  |     1 |        5 |         100    |                0    |          nan    |                   10    |
| outage_stall |     0 |        6 |         100    |               22.67 |          nan    |                    2.5  |
| outage_stall |     1 |        6 |         100    |                0    |           13.5  |                    1.67 |
| ramp_delay   |     0 |       42 |           0    |                0    |          955.25 |                    0    |
| ramp_delay   |     1 |       42 |           0    |                0    |          978.82 |                    0    |
| ramp_loss    |     0 |       54 |          17.59 |                0.24 |           93.35 |                    0.06 |
| ramp_loss    |     1 |       54 |          17.24 |                0    |          154.02 |                    0.13 |
| random_loss  |     0 |       11 |          16.68 |                0.92 |          116.45 |                    0.45 |
| random_loss  |     1 |       11 |          17.28 |                0    |          132.54 |                    0.45 |
