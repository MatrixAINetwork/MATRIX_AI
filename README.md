### WEEKLY UPDATES on MATRIX SMARTCONTRACT PROJECT

11th June - 15th June, 2018

##### Semantic Computing


- Discuss and research on how to achieve auto extraction of contract semantics 

- Propose a high-level design on auto extraction of contract semantics based on contract CFG (Control Flow Graph)

- Employ EVM bytecode parsing framework to generate CFG and automatically extract eigenvectors, as well as execute simple tests

- Finalized a Semantic Computing blueprint which is based on contract eigenvectors, including algorithm design, third-party libraries selection, the building of developing enviromnets, etc. Semantic Computing is currently in dev phase


##### Upgradable and Safe VM

- Discuss and research on how to address VM architecture and implementation details; go through go-ethereum VM code structures and dev interfaces;

- Finalize the high-level safe upgrade design for the VM end

- Based on the VM implementations of go-ethereum1.8.11, we completed the extensions to core module of interpreter, as well as JUMP_TABLE mapping

- Introduced a "Patched Contract" support to the VM, and added a —patchfile command line option;

- Prepared use cases for regression testing on new features;

- Completed a demo version of "Patched Contract" on VM side against BEC/SMT overflow vulnerbility.


18th June - 22th June, 2018

##### Semantic Computing

- Achieve the goal of automatic extraction of contract semantics characteristics as regards the EVM bytecode instructions; tests also done;

- Completed the training process to the GAlib-based semantic computing model, and performed the fitting training of simple data set for the model parameters;

- Completed framework codes for contract semantics clustering; finalized the use of third party libraries and the format of interactive data;

- Carrying out the analysis work on contract semantics template ; omnichannel preparations for demo use; as well as result analysis.


##### Upgradable and Safe VM

- Go on with the development of "Patched Contract", and go through code structures based on step-by-step debugging;