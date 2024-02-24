
# parsetab.py
# This file is automatically generated. Do not edit.
# pylint: disable=W,C,R
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'AND AS BATCH_SIZE BOOL BY CLONE CREATE DEBUG DELIMITER EPOCH ESTIMATOR FLOAT FORMULA FORMULA_EXP INT LEARNING_RATE LOSS MODEL OPTIMIZER PREDICT REGULARIZER SET SHUFFLE SQL TEST TRAIN TRAINING_PROFILE TYPE URL USE VALIDATION_SPLIT WEIGHTS WITH WORDexp : CREATE ESTIMATOR WORD TYPE WORD FORMULA FORMULA_EXP DELIMITER\n            | CREATE ESTIMATOR WORD TYPE WORD FORMULA FORMULA_EXP LOSS WORD DELIMITER\n            | CREATE ESTIMATOR WORD TYPE WORD FORMULA FORMULA_EXP LOSS WORD LEARNING_RATE FLOAT DELIMITER\n            | CREATE ESTIMATOR WORD TYPE WORD FORMULA FORMULA_EXP LOSS WORD LEARNING_RATE FLOAT OPTIMIZER WORD REGULARIZER WORD DELIMITERexp : CREATE TRAINING_PROFILE WORD WITH SQL DELIMITER\n                | CREATE TRAINING_PROFILE WORD WITH SQL AND VALIDATION_SPLIT FLOAT DELIMITER\n                | CREATE TRAINING_PROFILE WORD WITH SQL AND VALIDATION_SPLIT FLOAT BATCH_SIZE INT EPOCH INT DELIMITER\n                | CREATE TRAINING_PROFILE WORD WITH SQL AND VALIDATION_SPLIT FLOAT BATCH_SIZE INT EPOCH INT SHUFFLE BOOL DELIMITERexp : TRAIN WORD WITH WORD DELIMITER\n           | TRAIN WORD WITH TRAINING_PROFILE WORD DELIMITERexp : PREDICT WITH SQL BY ESTIMATOR WORD DELIMITER\n           | PREDICT WITH TRAINING_PROFILE WORD BY ESTIMATOR WORD DELIMITERexp : CLONE ESTIMATOR WORD AS WORD DELIMITER\n           | CLONE ESTIMATOR WORD AS WORD WITH WEIGHTS DELIMITERexp : USE URL DELIMITERexp : SQL DELIMITERexp : SET DEBUG BOOL DELIMITER'
    
_lr_action_items = {'CREATE':([0,],[2,]),'TRAIN':([0,],[4,]),'PREDICT':([0,],[5,]),'CLONE':([0,],[6,]),'USE':([0,],[7,]),'SQL':([0,13,26,],[3,20,34,]),'SET':([0,],[8,]),'$end':([1,11,23,32,35,41,43,46,50,53,56,57,59,61,66,70,75,76,],[0,-16,-15,-17,-9,-5,-10,-13,-11,-1,-12,-14,-6,-2,-3,-7,-8,-4,]),'ESTIMATOR':([2,6,29,38,],[9,14,37,45,]),'TRAINING_PROFILE':([2,13,19,],[10,21,28,]),'DELIMITER':([3,15,24,27,34,36,39,44,48,51,52,55,58,64,68,73,74,],[11,23,32,35,41,43,46,50,53,56,57,59,61,66,70,75,76,]),'WORD':([4,9,10,14,19,21,25,28,31,37,45,54,67,72,],[12,17,18,22,27,30,33,36,39,44,51,58,69,74,]),'WITH':([5,12,18,39,],[13,19,26,47,]),'URL':([7,],[15,]),'DEBUG':([8,],[16,]),'BOOL':([16,71,],[24,73,]),'TYPE':([17,],[25,]),'BY':([20,30,],[29,38,]),'AS':([22,],[31,]),'FORMULA':([33,],[40,]),'AND':([34,],[42,]),'FORMULA_EXP':([40,],[48,]),'VALIDATION_SPLIT':([42,],[49,]),'WEIGHTS':([47,],[52,]),'LOSS':([48,],[54,]),'FLOAT':([49,62,],[55,64,]),'BATCH_SIZE':([55,],[60,]),'LEARNING_RATE':([58,],[62,]),'INT':([60,65,],[63,68,]),'EPOCH':([63,],[65,]),'OPTIMIZER':([64,],[67,]),'SHUFFLE':([68,],[71,]),'REGULARIZER':([69,],[72,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'exp':([0,],[1,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> exp","S'",1,None,None,None),
  ('exp -> CREATE ESTIMATOR WORD TYPE WORD FORMULA FORMULA_EXP DELIMITER','exp',8,'p_create_model','parser.py',43),
  ('exp -> CREATE ESTIMATOR WORD TYPE WORD FORMULA FORMULA_EXP LOSS WORD DELIMITER','exp',10,'p_create_model','parser.py',44),
  ('exp -> CREATE ESTIMATOR WORD TYPE WORD FORMULA FORMULA_EXP LOSS WORD LEARNING_RATE FLOAT DELIMITER','exp',12,'p_create_model','parser.py',45),
  ('exp -> CREATE ESTIMATOR WORD TYPE WORD FORMULA FORMULA_EXP LOSS WORD LEARNING_RATE FLOAT OPTIMIZER WORD REGULARIZER WORD DELIMITER','exp',16,'p_create_model','parser.py',46),
  ('exp -> CREATE TRAINING_PROFILE WORD WITH SQL DELIMITER','exp',6,'p_training_profile','parser.py',94),
  ('exp -> CREATE TRAINING_PROFILE WORD WITH SQL AND VALIDATION_SPLIT FLOAT DELIMITER','exp',9,'p_training_profile','parser.py',95),
  ('exp -> CREATE TRAINING_PROFILE WORD WITH SQL AND VALIDATION_SPLIT FLOAT BATCH_SIZE INT EPOCH INT DELIMITER','exp',13,'p_training_profile','parser.py',96),
  ('exp -> CREATE TRAINING_PROFILE WORD WITH SQL AND VALIDATION_SPLIT FLOAT BATCH_SIZE INT EPOCH INT SHUFFLE BOOL DELIMITER','exp',15,'p_training_profile','parser.py',97),
  ('exp -> TRAIN WORD WITH WORD DELIMITER','exp',5,'p_train','parser.py',127),
  ('exp -> TRAIN WORD WITH TRAINING_PROFILE WORD DELIMITER','exp',6,'p_train','parser.py',128),
  ('exp -> PREDICT WITH SQL BY ESTIMATOR WORD DELIMITER','exp',7,'p_predict','parser.py',149),
  ('exp -> PREDICT WITH TRAINING_PROFILE WORD BY ESTIMATOR WORD DELIMITER','exp',8,'p_predict','parser.py',150),
  ('exp -> CLONE ESTIMATOR WORD AS WORD DELIMITER','exp',6,'p_clone_model','parser.py',175),
  ('exp -> CLONE ESTIMATOR WORD AS WORD WITH WEIGHTS DELIMITER','exp',8,'p_clone_model','parser.py',176),
  ('exp -> USE URL DELIMITER','exp',3,'p_use_database','parser.py',199),
  ('exp -> SQL DELIMITER','exp',2,'p_SQL','parser.py',214),
  ('exp -> SET DEBUG BOOL DELIMITER','exp',4,'p_DEBUG','parser.py',222),
]
