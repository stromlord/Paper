function [f_I1, f_I2, f_I3] = Freq_Intensity(f_I0, f_I1, f_I2, f_I3)

standard1 = max(max(f_I0))./max(max(f_I1));
standard2 = max(max(f_I0))./max(max(f_I2));
standard3 = max(max(f_I0))./max(max(f_I3));

f_I1 = f_I1.*standard1;
f_I2 = f_I2.*standard2;
f_I3 = f_I3.*standard3;


