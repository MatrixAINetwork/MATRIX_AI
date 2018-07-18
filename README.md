### WEEKLY UPDATES on MATRIX SMARTCONTRACT PROJECT

## 11th June - 15th June, 2018

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


## 18th June - 22th June, 2018

##### Semantic Computing

- Achieve the goal of automatic extraction of contract semantics characteristics as regards the EVM bytecode instructions; tests also done;

- Completed the training process to the GAlib-based semantic computing model, and performed the fitting training of simple data set for the model parameters;

- Completed framework codes for contract semantics clustering; finalized the use of third party libraries and the format of interactive data;

- Carrying out the analysis work on contract semantics template ; omnichannel preparations for demo use; as well as result analysis.


##### Upgradable and Safe VM

- Go on with the development of "Patched Contract", and go through code structures based on step-by-step debugging;

- Completed the development work of Transaction Function Recognition, and extended on the map data structure in statedb;

- Determined the implementation plan of patch permission authentication mechanism, focusing on the authentication scheme based on account address;

- Conducted research on contract upgrade, existing technology roadmap of patches, as well as coding design patterns;

- Built up a private chain for testing purpose, and completed the application of contract patches on the private chain


## 25th June - 29th June, 2018

##### Semantic Computing

- Completed the automatic framework of Semantic Computing, including characteristics extraction, model training and clustering;

- Complated the crawling and preparation of datasets;

- Optimized the training process to the GAlib-based semantic computing model, and completed the fitting training of contract code dataset for the  model parameters;

- Completed the code similarity calculation algorithm based on Bipartie Matching algorithm;

- Completed the premilinary real contract data semantics clustering with not so ideal results; result analysis and extended experiments are in progress;

- Discussed together on the plan of framework optimization 

##### Upgradable and Safe VM

- Research and discuss on contract upgrade plans and come out with a report; discuss on limitations and application scenarios of each upgrade plan;
- Go on with the development of "Patched Contract", refactor the code implementations under Core package; go through code structures based on step-by-step debugging;
- Implement refactoring and optimizations on VM codes deployed and upgraded on private chain;
- Read through go-ethereum API documetns and principles, as well as web3 related documents;
- Get down with technical proposal design on upgradable contract and VM


## 3rd July - 7th July, 2018

##### Semantic Computing

- Completed the semantics analysis on the current token based or walled based contract, and summarized several typical application scenarios;

- Completed design on automatically generated interaction scenarios by smart contracts;

- Completed the framework implementation of cluster based smartcontract semantics computing tools;

- Achieved the optimized extraction of semantic information, and completed the automatic training of weighting parameter based on GAlib;

##### Upgradable and Safe VM

- Discuss and research on the design scheme of Fault Tolerance system, analysing the practicability from the perspectives of technology and application of this proposal

- Go on with the development of "Patched Contract" supporting VM, and completed the optimized implementation of the PC address redirecting;

- Regarding the security vulnerability prediction engine related work, we focused on the taint analysis technology, and analyzed the feasibility of this technology at the virtual machine level;

- Completed the merge of open-source taint analysis and patching;

- Performed testing and debugging on security vulnerability prediction functions in private test-net.


## 9rd July - 13rd July, 2018

##### Semantic Computing


- Completed the design refination of the interactive scenarios generated automatically by smartcontracts, and finalized the prototype in form of web service, through different discussions;

- Frontend & background design and development in progress as regards smartcontract auto-generated prototype application scenarios 

- Further research on existing contracts, and refine the feature classification and induction of application scenarios


- Completed the semantic extraction optimization plan of automatic contract semantics clustering tools, inlucding the optimization of original semantic vectors, computing models, etc;

- Completed the extention of dataset for evaluation and preparation of dataset; fine-tuned the parameters of automatic semantic clustering model;

- Evaluating the effects of clustering based on full datasets.


##### Upgradable and Safe VM