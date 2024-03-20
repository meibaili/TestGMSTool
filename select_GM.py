import numpy as np
import timeit
import matplotlib.pyplot as plt
# from Function_CalculateInterPeriodCorr import * # Meibai commented out, since it is not used
import re
import pandas as pd
import itertools
from sklearn.covariance import LedoitWolf
import time
import multiprocessing
from functools import partial
import os



plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'font.family': 'DejaVu Sans'})

class search_ground_motion_CMS():
    def __init__(self):
        ### Assing the user inputs to their associated variables ###
        self.InputFolder_ResponseSpectrum = './Inputs'
        self.SaveFolderPath = './Outputs'
        self.DatasetsFolderPath = './Datasets'
        self.targetSpectraFileNames = ['TargetSpectra_Elastic']#,'TargetSpectra_ductility2.0',
                                       #'TargetSpectra_ductility3.0', 'TargetSpectra_ductility4.0']
        self.only_elastic = False
        self.InputVs = 760
        self.MScenario = None
        self.Msd = 1.5
        self.Mmin = None
        self.RMax = None #The largest distance for the record to be selected
        self.nGM =  20  #Number of records to be selected
        self.nPulse = None  #Number of pulse records to be selected
        self.TPulsemin = None #The lowest limit of pulse period of pulse records to be selected
        self.DuraMin = None #The smallest duration of the records to be selected
        self.periodCondition   =  4.0   #If the CMS formulation is used
        self.imCondition = 'Elastic'
        self.MaxNoGMsFromOneEvent =  3  #Maximum number of records that could come from a single event
        self.number_sd=2.5
        ### Other inputs ###
        self.databaseFileNames = ['Horizontal_elastic_PSA']#,'Horizontal_inelastic_2',
                                  #'Horizontal_inelastic_3','Horizontal_inelastic_4']
        self.Attrs = 'Horizontal_elastic_Attrs'
        self.maxScale=4 #Maximum scale to be used in record selection
        self.minScale=0.25 #Minimum scale to be used in record selection
        self.Dif_Vs_log = 1 # Maximum difference between Vs30 of site and the Vs30 of ground motion
        self.noScale=50 #50 scale factors from 0.5 to maxScal
        self.booleanFigure = False
        

    def run_ground_motion_selection(self):        
        self.read_inputs()
        self.scale_all()
        self.subset()
        self.select_GM_greedy()
        #self.add_moments_elastic()
        self.make_report_selection()
        
        
    def read_inputs(self):
        target_spectra_array = []
        Database_array = []
        for target_name,database_name in zip(self.targetSpectraFileNames,self.databaseFileNames):
            target_spectra_one = pd.read_csv(self.InputFolder_ResponseSpectrum + "/" + target_name + ".csv")
            type_IM_target = target_name.split('_')[1]
            target_spectra_one['T'] = [str(t)+'_'+type_IM_target for t in target_spectra_one['T']]
            target_spectra_array.append(target_spectra_one)
            Database_one = pd.read_csv(self.DatasetsFolderPath+"/"+database_name+'.csv',index_col=0)
            Database_one.columns = [a+'_'+type_IM_target for a in Database_one.columns]
            Database_array.append(Database_one)
        self.target_spectra = pd.concat(target_spectra_array,ignore_index=True)
        #print('Minimum of sd')
        #print(min(self.target_spectra['Sd']))
        if min(self.target_spectra['Sd'])==0:
            T = self.target_spectra['T'][np.where(self.target_spectra['Sd']==0)[0]].values[0]
            print(T)
            self.imCondition = T.split('_')[1]
            self.periodCondition = float(T.split('_')[0])
            self.CS_std = True
            print('Nice! the code understood there is pinch')
        else:
            self.CS_std = False
        Database = pd.concat(Database_array,axis=1, join='inner')
        Attrs = pd.read_csv(self.DatasetsFolderPath+"/"+self.Attrs+'.csv',index_col=0)
        index = Database.index
        self.Attrs = Attrs.loc[index,:].copy()
        periods2keep = []
        periods_target = self.target_spectra['T'].values
        periods_database = Database.columns
        periods2keep=[]
        for period_target in periods_target:
            period_found=False
            period_float_target = float(period_target.split('_')[0])
            type_IM_target = period_target.split('_')[1]
            for period_database in periods_database:
                period_float_database = float(period_database.split('_')[0])
                type_IM_database = period_database.split('_')[1]
                if abs( period_float_database - period_float_target )<0.001 and type_IM_target == type_IM_database:
                    periods2keep.append(period_database)
                    period_found=True
                    break
            if not period_found:
                print('Period of input does not match period of spectral accelerations in database')
                print('Handling this is not implemented yet')             
        self.Database = Database.loc[:,periods2keep].copy()

    def scale_all(self):
        periodIm = self.target_spectra['T'].values
        if self.CS_std:
            type_IM = [a.split('_')[1] for a in periodIm]
            period_float = np.array([float(a.split('_')[0]) for a in periodIm])
            condition1 = [t==self.imCondition for t in type_IM]
            index_condition = (abs(period_float  - self.periodCondition)<0.001)*(condition1)
            index_condition = np.where(index_condition)[0][0]
            print('Index of condition')
            print(index_condition)
            scale_all = self.target_spectra['Mean'].iloc[index_condition]/self.Database.iloc[:,index_condition].values
            self.Attrs['scale'] = scale_all
            self.Database = self.Database.multiply(scale_all, axis=0)
            if self.booleanFigure:
                types_IM_unique = np.unique(type_IM)
                for type_IM_unique in types_IM_unique:
                    plt.figure()
                    condition = [t==type_IM_unique for t in type_IM]
                    plt.loglog(period_float[condition],self.target_spectra.loc[condition,'Mean'],color='red')
                    for index,row in self.Database.iterrows():         
                        if index<50:
                            plt.loglog(period_float[condition],row.iloc[condition],color='gray')
                plt.show()
        else:
            scales = np.geomspace(self.minScale,self.maxScale,self.noScale)
            array_data_frame = []
            array_Att_data_frame = []
            for scale in scales:
                data_spectra_single_scaled = self.Database*scale
                multiindex_spectra = pd.MultiIndex.from_product([list(data_spectra_single_scaled.index), [scale]])
                data_spectra_single_scaled.set_index(multiindex_spectra,inplace=True)
                array_data_frame.append(data_spectra_single_scaled)
                Att_single_scale = self.Attrs.copy()
                Att_single_scale['scale']=scale
                multiindex_Att = pd.MultiIndex.from_product([list(Att_single_scale.index), [scale]])
                Att_single_scale.set_index(multiindex_Att,inplace=True)
                array_Att_data_frame.append(Att_single_scale)
            self.Database = pd.concat(array_data_frame)
            self.Database.sort_index(inplace=True)
            self.Attrs = pd.concat(array_Att_data_frame)
            self.Attrs.sort_index(inplace=True)        

    def subset(self):
        self.Attrs = self.Attrs[(self.Database > 0).all(1)].copy()
        self.Database = self.Database[(self.Database > 0).all(1)].copy()
        Dif_Sa = (np.log(self.Database) - np.log(self.target_spectra['Mean'].values))**2/self.target_spectra['Sd'].values
        Dif_Sa.replace(np.inf,0,inplace=True)
        #print(Dif_Sa)
        maxValues = Dif_Sa.max(axis = 1)
        indexCloseEnough = (maxValues<self.number_sd)
        scaleIsFine = (self.Attrs['scale']<self.maxScale)*(self.Attrs['scale']>self.minScale)
        indexCloseEnough *= scaleIsFine
        periodIm = self.target_spectra['T'].values
        period_float = np.array([float(a.split('_')[0]) for a in periodIm])
        maximum_target_period = max(period_float)
        minimum_target_frequency = 1/maximum_target_period
        filterFreqIsFine = self.Attrs['Lowest Usable Freq - Ave. Component (Hz)']<minimum_target_frequency
        indexCloseEnough *= filterFreqIsFine
        if self.MScenario:
            if self.Mmin:
                M_lower = max(self.Mmin, (self.MScenario - self.Msd)) 
            else:
                M_lower = self.MScenario - self.Msd           
            MIsFine = (self.Attrs['Magnitude']>M_lower)
            indexCloseEnough *= MIsFine
        if self.RMax:
            RIsFine = (self.Attrs['Rjb']<self.RMax)
            indexCloseEnough *= RIsFine
        if self.nPulse:
            if self.TPulsemin:
                PulseIsFine = (self.Attrs['Tp']==-999) | (self.Attrs['Tp']>=self.TPulsemin)
                indexCloseEnough *= PulseIsFine
        if self.DuraMin:
            DuraIsFine = ~((self.Attrs['Tp']<3.0) & (self.Attrs['Duration_5_95']<self.DuraMin))
            indexCloseEnough *= DuraIsFine
        if self.booleanFigure:
            periodIm = self.target_spectra['T'].values
            type_IM = [a.split('_')[1] for a in periodIm]
            period_float = np.array([float(a.split('_')[0]) for a in periodIm])
            types_IM_unique = np.unique(type_IM)
            for type_IM_unique in types_IM_unique:
                condition = [t==type_IM_unique for t in type_IM]
                plt.figure()
                plt.title(len(self.Database.loc[indexCloseEnough,]))
                plt.loglog(period_float[condition],self.Database.loc[indexCloseEnough,condition].T,color='gray')
        self.Database = self.Database.loc[indexCloseEnough,].copy()
        self.Attrs = self.Attrs.loc[indexCloseEnough,].copy()
        assert (self.Database.values > 0).any()

    def select_GM_greedy(self):
        if self.CS_std:
            periodIm = self.target_spectra['T'].values
            type_IM = [a.split('_')[1] for a in periodIm]
            period_float = np.array([float(a.split('_')[0]) for a in periodIm])
            condition1 = [t==self.imCondition for t in type_IM]
            index_condition = (abs(period_float  - self.periodCondition)<0.001)*(condition1)
            index_condition = np.where(index_condition)[0][0]
            data_temp = self.Database.drop(self.Database.columns[index_condition], axis=1)
            target_temp = self.target_spectra.drop(self.target_spectra.index[index_condition])
        else:
            data_temp = self.Database.copy()
            target_temp = self.target_spectra.copy()    
        if self.only_elastic:
            periodIm = target_temp['T'].values
            type_IM = [a.split('_')[1] for a in periodIm]
            condition1 = [t=='Elastic' for t in type_IM]
            data_temp = data_temp.loc[:,condition1]
            target_temp = target_temp.loc[condition1,:]

        sd1 = target_temp['Sd'].values
        mu1 = np.log(target_temp['Mean'].values)
        self.index_selected = []
        Attrs_search_into = self.Attrs.copy()
        for i in range(self.nGM):
            cost_all = []
            GMs = Attrs_search_into.index
            for GM in GMs:
                index_selected_temp = self.index_selected.copy()
                index_selected_temp.append(GM)
                cost_all.append(self.get_cost(index_selected_temp,data_temp,self.Attrs,sd1,mu1))
            index_min = np.argmin(cost_all)
            self.index_selected.append(GMs[index_min])
            if isinstance(GMs[index_min], tuple):
                Attrs_search_into.drop(index=GMs[index_min][0],inplace=True)
            else:
                Attrs_search_into.drop(index=GMs[index_min],inplace=True)
            #print(self.index_selected)
        changed_in_a_loop=True
        while changed_in_a_loop:
            changed_in_a_loop=False
            for j in range(self.nGM):
                cost_current = self.get_cost(self.index_selected,data_temp,self.Attrs,sd1,mu1)
                index_selected_delete_jth = self.index_selected.copy()
                index_selected_delete_jth.remove(index_selected_delete_jth[j])
                cost_all = []
                GMs = Attrs_search_into.index
                for GM in GMs:
                    index_selected_temp = index_selected_delete_jth.copy()
                    index_selected_temp.append(GM)
                    cost_all.append(self.get_cost(index_selected_temp,data_temp,self.Attrs,sd1,mu1))
                if min(cost_all)<cost_current:
                    changed_in_a_loop=True
                    index_min = np.argmin(cost_all)
                    self.index_selected[j] = GMs[index_min]
                    if isinstance(GMs[index_min], tuple):
                        Attrs_search_into.drop(index=GMs[index_min][0],inplace=True)
                    else:
                        Attrs_search_into.drop(index=GMs[index_min],inplace=True)
                    #print(self.index_selected)
        if self.nPulse:
            temp_Attrs = self.Attrs.loc[self.index_selected,:]
            nPulseSelected = len(temp_Attrs[temp_Attrs['Tp']!=-999])
            if nPulseSelected  < self.nPulse:
                Attrs_search_into_pulse = Attrs_search_into[Attrs_search_into['Tp']!=-999].copy()
                for k in range(self.nPulse - nPulseSelected):
                    # Replace every non-pulse record with a pulse record and find the one resulted in smallest cost
                    cost_all = []
                    for m in temp_Attrs[temp_Attrs['Tp']==-999].index:
                        index_selected_delete_mth = self.index_selected.copy()
                        index_selected_delete_mth.remove(m)

                        GMs = Attrs_search_into_pulse.index
                        for GM in GMs:
                            index_selected_temp = index_selected_delete_mth.copy()
                            index_selected_temp.append(GM)
                            cost_all.append([self.get_cost(index_selected_temp,data_temp,self.Attrs,sd1,mu1), m, GM])

                    # Find the m and GM that produce the smallest cost:
                    index_min = min(range(len(cost_all)), key=lambda n: cost_all[n][0])
                    m_min = cost_all[index_min][1]  # Record to be dropped from index_selected
                    GM_min = cost_all[index_min][2]  # Record to be added to index_selected
                    self.index_selected.remove(m_min)
                    self.index_selected.append(GM_min)

                    # Update temp_Attrs and Attrs_search_into_pulse
                    temp_Attrs = self.Attrs.loc[self.index_selected,:]
                    if isinstance(GM_min, tuple):
                        Attrs_search_into_pulse.drop(index=GM_min[0],inplace=True)
                    else:
                        Attrs_search_into_pulse.drop(index=GM_min,inplace=True)
                    #print(self.index_selected)
                    
            elif nPulseSelected  > self.nPulse:
                Attrs_search_into_pulse = Attrs_search_into[Attrs_search_into['Tp']==-999].copy()
                for k in range(nPulseSelected - self.nPulse):
                    # Replace every pulse record with a non-pulse record and find the one resulted in smallest cost
                    cost_all = []
                    for m in temp_Attrs[temp_Attrs['Tp']!=-999].index:
                        index_selected_delete_mth = self.index_selected.copy()
                        index_selected_delete_mth.remove(m)

                        GMs = Attrs_search_into_pulse.index
                        for GM in GMs:
                            index_selected_temp = index_selected_delete_mth.copy()
                            index_selected_temp.append(GM)
                            cost_all.append([self.get_cost(index_selected_temp,data_temp,self.Attrs,sd1,mu1), m, GM])

                    # Find the m and GM that produce the smallest cost:
                    index_min = min(range(len(cost_all)), key=lambda n: cost_all[n][0])
                    m_min = cost_all[index_min][1]  # Record to be dropped from index_selected
                    GM_min = cost_all[index_min][2]  # Record to be added to index_selected
                    self.index_selected.remove(m_min)
                    self.index_selected.append(GM_min)

                    # Update temp_Attrs and Attrs_search_into_pulse
                    temp_Attrs = self.Attrs.loc[self.index_selected,:]
                    if isinstance(GM_min, tuple):
                        Attrs_search_into_pulse.drop(index=GM_min[0],inplace=True)
                    else:
                        Attrs_search_into_pulse.drop(index=GM_min,inplace=True)
                    #print(self.index_selected)
           

    
    def get_cost(self, subset,Sa,Attrs,sd1,mu1):
        maxGmOneEq=self.MaxNoGMsFromOneEvent
        if max(Attrs.loc[subset,'EQID'].value_counts())>maxGmOneEq:
            return 10**6
        x = np.log(Sa.loc[subset,:].values)
        if len(Attrs.loc[subset,'EQID'])>3:
            x_bar = np.nanmean(x,axis=0)
            x_sd = np.nanmean((x-x_bar)**2,axis=0)**(0.5)
            cost = np.sum((mu1-x_bar)**2/sd1**2+x_sd**2/sd1**2+2*np.log(sd1/x_sd))
            #cost = np.sum((mu1-x_bar)**2/sd1**2)
        elif len(Attrs.loc[subset,'EQID'])>1:
            x_bar = np.nanmean(x,axis=0)
            #cost = np.sum((mu1-x_bar)**2/sd1**2)
            cost = np.sum((mu1-x_bar)**2/sd1**2)
        else:
            #cost = np.sum((mu1-x)**2/sd1**2)
            cost = np.sum((mu1-x)**2/sd1**2)
            #cost = np.random.rand(1)
        return cost

    def make_report_selection(self):
        if not os.path.exists(self.SaveFolderPath):
            os.makedirs(self.SaveFolderPath)
        #print(self.Attrs.loc[self.index_selected,:])
        df_save = self.Attrs.loc[self.index_selected,:]
        df_save.index = df_save.index.set_names(['RSN', 'SF'])
        df_save = df_save.reset_index()
        cols_to_select = ['RSN', 'EQID', 'Magnitude', 'Rjb', 'Vs30', 'scale', 'Duration_5_95', 'Tp', 'EQname', 'StationName']
        df_save = df_save[cols_to_select]
        df_save.to_csv(self.SaveFolderPath+'/'+'selected_motions.csv', index=False)
        periodIm = self.target_spectra['T'].values
        type_IM = [a.split('_')[1] for a in periodIm]
        period_float = np.array([float(a.split('_')[0]) for a in periodIm])
        x = np.log(self.Database.loc[self.index_selected,:].values)
        x_bar = np.nanmean(x,axis=0)
        x_sd = np.nanmean((x-x_bar)**2,axis=0)**(0.5)
        error_mean = x_bar-np.log(self.target_spectra['Mean'])
        error_sd = x_sd-self.target_spectra['Sd']
        evaluation_dic = {'period':self.target_spectra['T'].values,
                          'error_mean':error_mean,
                          'error_sd':error_sd}
        evaluation_df = pd.DataFrame(evaluation_dic)
        #evaluation_df.to_csv(self.SaveFolderPath+'/'+'evaluation.csv')
        self.target_spectra['plus_sd'] = self.target_spectra['Mean']*np.exp(self.target_spectra['Sd'])
        self.target_spectra['minus_sd'] = self.target_spectra['Mean']/np.exp(self.target_spectra['Sd'])
        types_IM_unique = np.unique(type_IM)
        fig, axs = plt.subplots(nrows=len(types_IM_unique), ncols=2,figsize=(7.5,2.5*len(types_IM_unique)))
        axs = axs.ravel()
        for type_IM_unique,ax in zip(types_IM_unique,axs[::2]):
            condition = [t==type_IM_unique for t in type_IM]
            ax.plot(period_float[condition],self.Database.loc[self.index_selected,condition].T,color = 'gray')
            ax.lines[-1].set_label('selected GMs')
            ax.plot(period_float[condition],np.exp(x_bar+x_sd)[condition],color = 'red',linewidth=3.0)
            ax.plot(period_float[condition],np.exp(x_bar-x_sd)[condition],color = 'red',label = 'suite +/-'+r'$\sigma$',linewidth=3.0)
            ax.plot(period_float[condition],self.target_spectra.loc[condition,'Mean'],'-x',color='black',label = 'Target')
            ax.plot(period_float[condition],np.exp(x_bar[condition]),color = 'blue',label='Mean suite')
            ax.plot(period_float[condition],self.target_spectra.loc[condition,'plus_sd'],'--',color='green',label = ' target +/-'+r'$\sigma$',linewidth=3.0)
            ax.plot(period_float[condition],self.target_spectra.loc[condition,'minus_sd'],'--',color='green',linewidth=3.0)
            ax.set( ylabel =r'$C_{y}$',xscale = 'log',yscale = 'log',ylim=[5e-3,5])
            ax.text(0.80,0.85,type_IM_unique.replace('ductility',r'$\mu$ = '),transform=ax.transAxes)
        ax.set_xlabel('T (s)')
        handles, labels = ax.get_legend_handles_labels()
        lgd1 = fig.legend(handles, labels, bbox_to_anchor=[0.5, 1], loc='lower right', ncol = 2)
        for type_IM_unique,ax in zip(types_IM_unique,axs[1::2]):
            condition = [t==type_IM_unique for t in type_IM]
            ax.semilogx(period_float[condition],evaluation_df['error_mean'][condition],color='blue',label='Mean')
            ax.semilogx(period_float[condition],evaluation_df['error_sd'][condition],color='red',label='Standard deviation')
            ax.semilogx(period_float[condition],0/period_float[condition],color='black')
            ax.set_ylim([-0.2,0.2])
            ax.set_ylabel('Suite minus target in log', labelpad=-5)
            ax.text(0.80,0.85,type_IM_unique.replace('ductility',r'$\mu$ = '),transform=ax.transAxes)
        ax.set_xlabel('T (s)')
        handles, labels = ax.get_legend_handles_labels()
        lgd2 = fig.legend(handles, labels, bbox_to_anchor=[0.5, 1], loc='lower left', ncol = 2)
        plt.savefig(self.SaveFolderPath+'/'+'/Selected motions.png',bbox_extra_artists=(lgd1,lgd2), bbox_inches='tight')
        plt.close()
        if 'TargetSpectra_Elastic' in self.targetSpectraFileNames:
            fig, ax = plt.subplots(figsize=(4,4))
            type_IM_unique = 'Elastic'
            condition = [t==type_IM_unique for t in type_IM]
            Database_log = np.log(self.Database.loc[self.index_selected,condition])
            cov = LedoitWolf().fit(Database_log)
            cor = self.correlation_from_covariance(cov.covariance_)
            im=ax.matshow(cor,vmin=-1,vmax=1)
            plt.colorbar(im,ax=ax)
            labels = ["{:.2f}".format(float(a.split('_')[0])) for a in Database_log.select_dtypes(['number']).columns]
            plt.xticks(range(Database_log.select_dtypes(['number']).shape[1]), labels, rotation=90)
            plt.yticks(range(Database_log.select_dtypes(['number']).shape[1]), labels)
            plt.xlabel('T (s)')
            plt.ylabel('T (s)')
            plt.savefig(self.SaveFolderPath+'/'+'cor_selected motions.png', bbox_inches='tight')
            plt.close()


    def correlation_from_covariance(self,covariance):
        v = np.sqrt(np.diag(covariance))
        outer_v = np.outer(v, v)
        correlation = covariance / outer_v
        correlation[covariance == 0] = 0
        return correlation

    def add_moments_elastic(self):
        periodIm = self.Database.columns
        
        period_float = np.array([float(a.split('_')[0]) for a in periodIm])
        condition = [p.endswith('Elastic') for p in periodIm]
        m0_array = []
        m1_array = []
        m2_array = []
        x = np.log(period_float[condition])
        c = np.log(self.periodCondition)
        for index,row in self.Database.iterrows():
            y = row[condition]
            m1,m2 = self.calculate_moments(x,y,c)
            m1_array.append(m1)
            m2_array.append(m2)
        self.Attrs['m1'] = m1_array
        self.Attrs['m2'] = m2_array

    def calculate_moments(self,x,y,c):
        m1 = sum(y*x)/sum(y)
        m2 = sum(y*(x-m1)**2)/sum(y)
        return m1,m2

    
if __name__ == '__main__':
    GMS_analysis = search_ground_motion_CMS()
    GMS_analysis.periodCondition = 2
    GMS_analysis.only_elastic = True
    GMS_analysis.MScenario=6
    GMS_analysis.run_ground_motion_selection()
    
