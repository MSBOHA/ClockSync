Accuracy Comparison of Different Time Synchronization Methods


TimeError.log
                
// REAL 数据段（字段1-10）
 ns2s(getT1(REAL)), ns2s(getT2(REAL)),
 ns2s(getT3(REAL)), ns2s(getT4(REAL)),
 ns2us(getT2T1(REAL)), ns2us(getT3T4(REAL)),
 ns2us(getOffset(REAL)), ns2us(getCalculatedOffset(REAL)),
 ns2us(getDelay(REAL)), ns2s(getT4T1(REAL)),

// RAW时钟数据段（字段11-20）
 ns2s(getT1(RAW)), ns2s(getT2(RAW)),
 ns2s(getT3(RAW)), ns2s(getT4(RAW)),
 ns2us(getT2T1(RAW)), ns2us(getT3T4(RAW)),
 ns2us(getOffset(RAW)), ns2us(getCalculatedOffset(RAW)),
 ns2us(getDelay(RAW)), ns2s(getT4T1(RAW)),

// 元数据段（字段21,22-25）
 seqNo,
 ns2us(t1RealKernel - t1RealApp), ns2us(t2RealApp - t2RealKernel),
 ns2us(t3RealKernel - t3RealApp), ns2us(t4RealApp - t4RealKernel)
