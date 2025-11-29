# Issues

## Issue #01

Implement and test the reading of the time stamp using the funcion `FDwfAnalogInStatusTime`

**STATUS** added on 240816, solved with v.1.1.0.

## Issue #02

Verify whether it is possible/useful to transfer data as `uint16` using the `SDK` function `FDwfAnalogInSignalData16(HWDF,int,short*,int,int)`. In principle it should be significantly faster and useful during long acquisitions. This implies retrieving the conversion parameters with `FDwfAnalogInChannelRangeGet`
and `FDwfAnalogInChannelOffsetGet`.

**STATUS** added on 240817, pending

## Issue #03

Implement the digital oscilloscope

**STATUS** added on 240820, pending

## Issue #04

Implement the support for SPI communciation

**STATUS** added on 240820, pending


