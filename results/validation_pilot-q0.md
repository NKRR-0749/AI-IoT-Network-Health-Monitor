# Testbed validation: run `pilot-q0`

| phase       |   seconds |   injected_loss_% |   injected_delay_ms |   udp_loss_measured_% |   mqtt_msg_loss_% |   rtt_median_ms |   rtt_lost |   disconnects |
|:------------|----------:|------------------:|--------------------:|----------------------:|------------------:|----------------:|-----------:|--------------:|
| clean       |        30 |                 0 |                   0 |                  0    |                 0 |             1.9 |          0 |             0 |
| random_loss |        40 |                10 |                   0 |                  8.25 |                 0 |            36.8 |          0 |             0 |
| clean       |        20 |                 0 |                   0 |                  0    |                 0 |             2.3 |          0 |             0 |
| delay       |        30 |                 0 |                 300 |                  0    |                 0 |           650.2 |          0 |             0 |
| clean       |        20 |                 0 |                   0 |                  0    |                 0 |             1.8 |          0 |             0 |
| outage_drop |        15 |               100 |                   0 |                100    |               100 |           nan   |          0 |             5 |
| clean       |        30 |                 0 |                   0 |                  0    |                 0 |             2.2 |         10 |             0 |
