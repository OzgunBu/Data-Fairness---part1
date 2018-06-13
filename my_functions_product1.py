#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:15:42 2018

@author: bursaliogluozgun
"""
import pandas as pd
import numpy as np


import keras as ke
import keras.backend as K
from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.models import load_model
# model reconstruction from JSON:
from keras.models import model_from_json


import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
#==============================================================================
def Input_file_checking(filename):
    """ 
       This function checks if the input file exists
    """
    data_file_error = False

    
    
    try:
        #open and closing the log file
        fdata = open(filename,'r')
        fdata.close() 
    except IOError:
        print('Cannot open '+filename)
        data_file_error = True
        
        
#==============================================================================
def reading_data(path):
    your_data_df = pd.read_csv(path)
    
    #your_data_df['Prediction'] = your_data_df[' <=50K']
    
    all_columns = np.array(your_data_df.columns)
    print('Columns in your data are:\n ',all_columns)
    all_columns = set(all_columns.tolist())
    #print(type(all_columns))
    
    return your_data_df, all_columns

#==============================================================================
def input_from_user(input_str, warn_str, result_str, possible_set):
    
    valid_input = False
    while valid_input == False:
        answer = input(input_str)
        if answer not in possible_set:
            print(warn_str)
        else:
            valid_input = True
    print(result_str+"'{}'".format(answer))
    return answer
#==============================================================================
    
def deleting_columns(fields_to_delete,columns_can_omitted):
    not_to_consider_fields = []
    if fields_to_delete == 'Y':
    
        not_to_consider_fields = []
        valid_input = False
        no_more = False


        while (valid_input == False) or (no_more == False):
            answer = input('Enter a column, you want to omit or enter "no more" if you are done:')


            if answer == 'no more':
                valid_input = True
                no_more = True
            else:
                if answer not in columns_can_omitted:
                    print('Not a valid column')
                    #answer = input('Enter a column, you want to omit or enter "no more" if you are done:')
                else:
                    valid_input = True
                    #no_more = False
                    not_to_consider_field = answer
                    not_to_consider_fields.append(not_to_consider_field) 
                
        print('Columns to be omitted: {}'.format(not_to_consider_fields))
    else:
        print('All columns are kept.')
                
    return not_to_consider_fields
#==============================================================================
def Inputting_from_the_user(all_columns,your_data_df):
    ### Inputting user preference for Target Field
           
   remaining_columns = all_columns.copy()
   possible_set = remaining_columns
   print("Your possible target columns '{}'".format(possible_set))
   
   input_str = 'Which column is your target column?:'
   warn_str = 'Please enter a valid column'
   result_str = "Your target column is "
   target_field = input_from_user(input_str, warn_str, result_str, possible_set)
   
   ### Inputting user preference for cPrediction Field (Current Prediction Field)
   #### Do you have a current prediction? Y or N
   possible_set = ['Y','N']
   input_str = 'Do you have any column for your current prediction? Please enter Y or N:'
   warn_str = 'Please enter Y or N'
   result_str = "Do you have any column for your current prediction? You entered"
   cprediction_field_exists = input_from_user(input_str, warn_str, result_str, possible_set)
   
   #### If you have what is your current prediction column?
   cprediction_field = []
   remaining_columns = all_columns.copy()
   remaining_columns.remove(target_field)

   if cprediction_field_exists == 'Y':
        possible_set = remaining_columns
        print("Your possible current prediction columns '{}'".format(possible_set))
        input_str = 'Which column is your current prediction column?:'
        warn_str = 'Please enter a valid column'
        result_str = "Your current prediction column is "
        cprediction_field = input_from_user(input_str, warn_str, result_str, possible_set)
        remaining_columns.remove(cprediction_field)
        
   ### Inputting user preference for Target 0-1- labels
   possible_set = set(your_data_df[target_field].unique())
   print("Your prediction column possible values '{}'".format(possible_set))
   print('Please enter two different labels in the target field: ')

   input_str = 'Enter label 0:'
   warn_str = 'Please enter a valid label within target field'
   result_str = "Your target label 0 is "
   target_label0 = input_from_user(input_str, warn_str, result_str, possible_set)

   possible_set.remove(target_label0)

   input_str = 'Enter label 1:'
   warn_str = 'Please enter a valid lable within target field'
   result_str = "Your target label 1 is "
   target_label1 = input_from_user(input_str, warn_str, result_str, possible_set)
   
   ### Inputting user preference for Sensitive Attribute
   
   remaining_columns = all_columns.copy()
   remaining_columns.remove(target_field)

   if cprediction_field_exists == 'Y':
       remaining_columns.remove(cprediction_field) 

   possible_set = remaining_columns
   print("Your possible Sensitive Attribute columns '{}'".format(possible_set))
   input_str = 'Which column is your sensitive column?:'
   warn_str = 'Please enter a valid column'
   result_str = "Your sensitive attribute column is "
   sensitive_field = input_from_user(input_str, warn_str, result_str, possible_set)
    
    ### Inputting user preference for Sensitive 0-1 classes
   possible_set = set(your_data_df[sensitive_field].unique())


   print('unique values from the sensitive class:{}'.format(possible_set))
   print('Please enter two different classes in the sensisitve field: ')

   input_str = 'Enter class 0:'
   warn_str = 'Please enter a valid class within sensitive field'
   result_str = "Your sensitive attribute class 0 is "
   sensitive_class0 = input_from_user(input_str, warn_str, result_str, possible_set)

   possible_set.remove(sensitive_class0)

   input_str = 'Enter class 1:'
   warn_str = 'Please enter a valid class within sensitive field'
   result_str = "Your sensitive attribute class 1 is "
   sensitive_class1 = input_from_user(input_str, warn_str, result_str, possible_set)
    
    ### Do you want to omit any column from your data?
   possible_set = ['Y','N']
   input_str = 'Do you want to omit any column from your data? Please enter Y or N:'
   warn_str = 'Please enter Y or N'
   result_str = "Do you have any column for your current prediction? You entered"
   fields_to_delete = input_from_user(input_str, warn_str, result_str, possible_set)
    
   remaining_columns = all_columns.copy()
   remaining_columns.remove(target_field)

   if cprediction_field_exists == 'Y':
       remaining_columns.remove(cprediction_field) 

   columns_can_omitted = remaining_columns
    
   not_to_consider_fields = deleting_columns(fields_to_delete,columns_can_omitted)

   return cprediction_field_exists,cprediction_field,target_field, sensitive_field, your_data_df, not_to_consider_fields,sensitive_class0,sensitive_class1,target_label0,target_label1
#==============================================================================
   
def processing_data_frame(output_field, sensitive_field, your_data_df, not_to_consider_fields,class0,class1):
    
    #drop columns
    for item in not_to_consider_fields:
        if item != sensitive_field:
            your_data_df = your_data_df.drop(item, axis=1)
            #print(item,'----',sensitive_field)
           # print(item,your_data_df.columns)
    
    #check for missing values, do sth with them
    your_data_df = your_data_df.dropna()
    
    # only keep the row with sensitive field equal to class1, and class2
    your_data_df = your_data_df.loc[lambda df: df[sensitive_field].isin([class0, class1])]

    #create Y
    Y_df = your_data_df[output_field]
    
    #create Z
    Z_df = your_data_df[sensitive_field]
    
    #create X
    X_df = your_data_df.copy()
    #print(X_df.shape)
    X_df = X_df.drop(output_field, axis=1)
    #print(X_df.shape)


    if sensitive_field in not_to_consider_fields:
        X_df = X_df.drop(sensitive_field, axis=1)
        
       
    
    return X_df, Y_df, Z_df
    
#==============================================================================
def bias_checker_p_rule_bin(Z, Y):
    
    
    Y_Z_class0 = Y[Z == 0]
    Y0_Z_class0 = Y_Z_class0[Y_Z_class0 == 0]
    Y1_Z_class0 = Y_Z_class0[Y_Z_class0 == 1]
    
    Y_Z_class1 = Y[Z == 1]
    Y0_Z_class1 = Y_Z_class1[Y_Z_class1 == 0]
    Y1_Z_class1 = Y_Z_class1[Y_Z_class1 == 1]
    
    Y0Z0 = (Y0_Z_class0.shape[0])
    Y1Z0 = (Y1_Z_class0.shape[0])
    Z0 = Y0Z0 + Y1Z0
    
    Y0Z1 = (Y0_Z_class1.shape[0])
    Y1Z1 = (Y1_Z_class1.shape[0])
    Z1 = Y0Z1 + Y1Z1
    
    
    p_rule_for_Y0 = format(100*min([(Y0Z1/Z1)/(Y0Z0/Z0),(Y0Z0/Z0)/(Y0Z1/Z1)]),'.2f')
    p_rule_for_Y1 = format(100*min([(Y1Z1/Z1)/(Y1Z0/Z0),(Y1Z0/Z0)/(Y1Z1/Z1)]),'.2f')
    
    
    return p_rule_for_Y0,p_rule_for_Y1

#==============================================================================
def enforcing_binary_output_sensitive(Y_df,output_field,output_label0, output_label1,Z_df,sensitive_field,sensitive_class0, sensitive_class1):
    
    Ybin = (Y_df == output_label1).astype(int)
    Zbin = (Z_df == sensitive_class1).astype(int)
    
    
    return Ybin, Zbin

#==============================================================================
def read_process_data_output_bias(filename_str):   
    your_data_df, all_columns = reading_data(filename_str)  
    cprediction_field_exists,cprediction_field,target_field, sensitive_field, your_data_df, not_to_consider_fields,sensitive_class0,sensitive_class1,target_label0,target_label1 = Inputting_from_the_user(all_columns,your_data_df)
   
    X_df, Y_df, Z_df = processing_data_frame(target_field, sensitive_field, your_data_df, not_to_consider_fields,sensitive_class0,sensitive_class1)    
 
    print('target field:{}'.format(target_field))
    if cprediction_field_exists == 'Y':
        print('current prediction field:{}'.format(cprediction_field))
    
    print('target field:{} Label 0-1 {}, {}'.format(target_field,target_label0,target_label1 ))
    print('sensitive field:{} Class 0-1 {}, {}'.format(sensitive_field,sensitive_class0,sensitive_class1 ))
    print('not to consider fields',not_to_consider_fields)
   
    # let'see the filename
    result_fname = result_filename(filename_str,cprediction_field_exists,cprediction_field,target_field, sensitive_field, not_to_consider_fields,sensitive_class0,sensitive_class1,target_label0,target_label1 )
    print('result file: ', result_fname )
    
    ### First let's see how biased your data:
   
    Ybin, Zbin = enforcing_binary_output_sensitive(Y_df,target_field,target_label0, target_label1,Z_df,sensitive_field,sensitive_class0,sensitive_class1)
    p_rule_for_Y0,p_rule_for_Y1 = bias_checker_p_rule_bin(Zbin, Ybin)   
    print('Target field: p_rule_for_Y1',p_rule_for_Y1)

    ### First let's see how biased your current prediction:

    if cprediction_field_exists == 'Y':
        Ybin, Zbin = enforcing_binary_output_sensitive(Y_df,cprediction_field,target_label0, target_label1,Z_df,sensitive_field,sensitive_class0,sensitive_class1)
        p_rule_for_Y0,p_rule_for_Y1 = bias_checker_p_rule_bin(Zbin, Ybin)   
        print('Current prediction field: p_rule_for_Y1',p_rule_for_Y1)
        
    return X_df, Ybin, Zbin, result_fname     

#==============================================================================
def result_filename(path,cprediction_field_exists,cprediction_field,target_field, sensitive_field, not_to_consider_fields,sensitive_class0,sensitive_class1,target_label0,target_label1 ):
    result_fname = path + '_'
    result_fname = result_fname + 'CP_' + cprediction_field_exists +'_'
    if cprediction_field_exists == 'Y':
        result_fname = result_fname + cprediction_field
    
    result_fname = result_fname + 'T_' + target_field +'_'+ target_label0 +'_'+ target_label1 +'_'
    result_fname = result_fname + 'S_' + sensitive_field +'_'+ sensitive_class0 +'_'+ sensitive_class1 +'_'
    
    result_fname =  result_fname.replace(' ','_')
    result_fname =  result_fname.replace('/' ,'--')
    
    return result_fname


#==============================================================================
#==============================================================================
 # HIDE

class FairClassifier(object):
    
    def __init__(self, tradeoff_lambda,main_task_arch_json_string,adv_task_arch_json_string,pre_load_flag=True,main_task_trained_weight_file=None):
        self.tradeoff_lambda = tradeoff_lambda
        
        
        
        clf_net = self._create_clf_net(main_task_arch_json_string)
        adv_net = self._create_adv_net(adv_task_arch_json_string)
        
        clf_inputs = clf_net.input
        
        #adv_inputs = adv_net.input
    
        
        self._trainable_clf_net = self._make_trainable(clf_net)
        self._trainable_adv_net = self._make_trainable(adv_net)
        self._clf = self._compile_clf(clf_net)
        
        self._clf_w_adv = self._compile_clf_w_adv(clf_inputs, clf_net, adv_net)
        self._adv = self._compile_adv(clf_inputs, clf_net, adv_net)
        self._val_metrics = None
        self._fairness_metrics = None
        
        self.predict = self._clf.predict
        
    def _make_trainable(self, net):
        def make_trainable(flag):
            net.trainable = flag
            for layer in net.layers:
                layer.trainable = flag
        return make_trainable
    
    
    def _create_clf_net(self, main_task_arch_json_string):
        architecture = model_from_json(main_task_arch_json_string)
        return(architecture)
    
    
    def _create_adv_net(self, adv_task_arch_json_string):
        architecture = model_from_json(adv_task_arch_json_string)
        return(architecture)    
    
    def _compile_clf(self, clf_net):
        clf = clf_net
        self._trainable_clf_net(True)
        clf.compile(loss='binary_crossentropy', optimizer='adam')
        return clf
        
    def _compile_clf_w_adv(self, inputs, clf_net, adv_net):
    
        clf_w_adv = Model(inputs=[inputs], outputs=[clf_net(inputs)]+[adv_net(clf_net(inputs))])
        
        self._trainable_clf_net(True)
        self._trainable_adv_net(False)
        loss_weights = [1.]+[-self.tradeoff_lambda]
        
        clf_w_adv.compile(loss=['binary_crossentropy']*(len(loss_weights)), 
                          loss_weights=loss_weights,
                          optimizer='adam')
        return clf_w_adv
    
    def _compile_adv(self, inputs, clf_net, adv_net):
        adv = Model(inputs=[inputs], outputs=adv_net(clf_net(inputs)))
        self._trainable_clf_net(False)
        self._trainable_adv_net(True)
        adv.compile(loss=['binary_crossentropy'], optimizer='adam')
        return adv
    
    def _compute_class_weights(self, data_set):
        class_values = [0, 1]
        class_weights = []
    
        balanced_weights = compute_class_weight('balanced', class_values, data_set)
        class_weights.append(dict(zip(class_values, balanced_weights)))
    
        return class_weights
    
    def _compute_target_class_weights(self, y):
        class_values  = [0,1]
        balanced_weights =  compute_class_weight('balanced', class_values, y)
        class_weights = {'y': dict(zip(class_values, balanced_weights))}
        
        return class_weights
        
    def pretrain(self, x, y, z, epochs=10, verbose=0,pre_load_flag=True,main_task_trained_weight_file=None):
        
        if pre_load_flag == False:
            self._trainable_clf_net(True)
            self._clf.fit(x.values, y.values, epochs=epochs, verbose=verbose)
            self._trainable_clf_net(False)
            self._trainable_adv_net(True)
            class_weight_adv = self._compute_class_weights(z)
            self._adv.fit(x.values, z.values, class_weight=class_weight_adv, 
                      epochs=epochs, verbose=verbose)
        else:
    
            self._clf.load_weights(main_task_trained_weight_file)
    
    
            
            self._trainable_clf_net(False)
            self._trainable_adv_net(True)
            class_weight_adv = self._compute_class_weights(z)
            self._adv.fit(x.values, z.values, class_weight=class_weight_adv, 
                      epochs=epochs, verbose=verbose)
            
    def fit(self, x, y, z, validation_data=None, T_iter=250, batch_size=128,
            save_figs=False):
        
        
        if validation_data is not None:
            x_val, y_val, z_val = validation_data
    
        class_weight_adv = self._compute_class_weights(z)
        class_weight_clf_w_adv = [{0:1., 1:1.}]+class_weight_adv
        
        #self._val_metrics = pd.DataFrame()
        #self._fairness_metrics = [] #pd.DataFrame()  
        
        for idx in range(T_iter):
            #print(idx)
            #if validation_data is not None:
                #y_pred = pd.Series(self._clf.predict(x_val).ravel(), index=y_val.index)
                #self._val_metrics.loc[idx, 'ROC AUC'] = roc_auc_score(y_val, y_pred)
                #self._val_metrics.loc[idx, 'Accuracy'] = (accuracy_score(y_val, (y_pred>0.5))*100)
                
               
                #self._fairness_metrics.append(myFC.bias_checker_p_rule_bin((y_pred>0.5)*1.0,z_val))
                
                #display.clear_output(wait=True)
    
            # train adverserial
            self._trainable_clf_net(False)
            self._trainable_adv_net(True)
            self._adv.fit(x.values, z.values, batch_size=batch_size, 
                          class_weight=class_weight_adv, epochs=1, verbose=0)
            
            # train classifier
            self._trainable_clf_net(True)
            self._trainable_adv_net(False)
            indices = np.random.permutation(len(x))[:batch_size]
            #self._clf_w_adv.train_on_batch(x.values[indices], 
                                           #[y.values[indices]]+[z.values[indices]])
            self._clf_w_adv.train_on_batch(x.values[indices], 
                                           [y.values[indices]]+[z.values[indices]],
                                           class_weight=class_weight_clf_w_adv)   


