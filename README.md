# Cell segmentation using MATLAB

This program has the purpose to ultimately segment an image in order
to find the relative intensity of the signal in Channel 1 based on the
cell walls form Channel 2.
----------Requirements:-----------
-Latest version of MATLAB (R2017b)
-Computer with GPU capabilities (also GPU driver up to date (update CUDA))
For reference it was developed in 
a Mac OS Sierra with CUDA and these specs from gpuDevice
 
                      Name: 'NVIDIA GeForce GT 650M'
                     Index: 1
         ComputeCapability: '3.0'
            SupportsDouble: 1
             DriverVersion: 8
            ToolkitVersion: 8
        MaxThreadsPerBlock: 1024
          MaxShmemPerBlock: 49152
        MaxThreadBlockSize: [1024 1024 64]
               MaxGridSize: [1×3 double]
                 SIMDWidth: 32
               TotalMemory: 1.0734e+09
           AvailableMemory: 82386944
       MultiprocessorCount: 2
              ClockRateKHz: 900000
               ComputeMode: 'Default'
      GPUOverlapsTransfers: 1
    KernelExecutionTimeout: 1
          CanMapHostMemory: 1
           DeviceSupported: 1
            DeviceSelected: 1

