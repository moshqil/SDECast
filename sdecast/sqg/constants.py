"""Physical constants for the SQG system.

``data_std`` is the standardisation applied at training time (state = PV / 2660),
and ``scalefact`` converts PV to potential temperature.
"""
nx, ny = 64, 64
data_mean = [0.0, 0.0]
data_std = [2660, 2660]
scalefact = 0.003061224412462883

# One train-time unit is H_HOURS hours. With sample_length=2 the posterior bridges
# a 2 * 3 = 6 h window, which is what the released SQG checkpoint was trained on.
# Upstream was inconsistent here (some scripts defaulted to 6); 3 is the value
# training actually used.
H_HOURS = 3.0
