import numpy as np
import pandas as pd
import pickle
import time
from scipy.stats import lognorm
from scipy.stats import gamma
from os import path


# VIRAL LOAD FUNCTIONS


def VL(t,T3,TP,YP,T6):
    '''
    VL(t,T3,TP,YP,T6)
        Computes the scalar value of a hinge function with control points:
        (T3, 3)
        (TP, YP)
        (T6, 6)
    Returns zero whenever (1) t<T3 or (2) hinge(t) is negative.
    '''
    if t < T3:
        return 0
    if t < TP:
        return (t-T3)*(YP-3)/(TP-T3)+3
    return np.max([(t-TP)*(6-YP)/(T6-TP)+YP,0])

    
def get_trajectory(is_symptomatic=False,full=False):
    '''
    get_trajectory(is_symptomatic=False,full=False)
        Stochastically draws a viral load trajectory by randomly drawing control points.
    
    is_symptomatic allows separate models for symptomatic and asymptomatic trajectories which
    the review by Cevik et al 2020 has shown to be different. (Asymptomatic clearance is faster)
    
    When full==False, returns a viral load trajectory evaluated on int days 0,1,2,...,27,
    where day 0 is the day of exposure. 

    When full==True, returns the integer-evaluated trajectory, as well as:
    v_fine the same trajectory evaluated at values between 0 and 28, with step size 0.01
    t_fine the steps at which the trajectory was evaluated
    t_3, t_peak, v_peak, t_6, the four stochastically drawn control values

    Use full==True for plotting; use full==False for more rapid simulation.
    '''
    if is_symptomatic==True:
        return get_symptomatic_trajectory(full=full)
    else:
        return get_asymptomatic_trajectory(full=full)


def get_symptomatic_trajectory(full=False):
    # Draw control points. 
    # Truncate so that max of t_peak is 3.
    t_peak = gamma.rvs(1.5,loc=0.5)
    while t_peak > 3:
        t_peak = gamma.rvs(1.5,loc=0.5)
    v_peak = np.random.uniform(7,11)
    t_6 = np.random.uniform(4,9)
    t_3 = np.random.random()+2.5
    t_symptoms = np.random.uniform(0,3)
    # Compute t and v vectors and return 
    t = np.arange(28)
    v = np.array([VL(x,t_3,t_peak+t_3,v_peak,t_6+t_3+t_peak+t_symptoms) for x in t])
    if full==True:
        tfine = np.arange(0,28,0.01)
        vfine = np.array([VL(x,t_3,t_peak+t_3,v_peak,t_6+t_3+t_peak+t_symptoms) for x in tfine])
        return v,vfine,tfine,t_3,t_peak,v_peak,t_6,t_symptoms
    return v,int(np.round(t_symptoms+t_3+t_peak))


def get_asymptomatic_trajectory(full=False):    
    # Draw control points. 
    # Truncate so that max of t_peak is 3.
    t_peak = gamma.rvs(1.5,loc=0.5)
    while t_peak > 3:
        t_peak = gamma.rvs(1.5,loc=0.5)
    v_peak = np.random.uniform(7,11)
    t_6 = np.random.uniform(4,9)
    t_3 = np.random.random()+2.5
    # Compute t and v vectors and return 
    t = np.arange(28)
    v = np.array([VL(x,t_3,t_peak+t_3,v_peak,t_6+t_3+t_peak) for x in t])
    if full==True:
        tfine = np.arange(0,28,0.01)
        vfine = np.array([VL(x,t_3,t_peak+t_3,v_peak,t_6+t_3+t_peak) for x in tfine])
        return v,vfine,tfine,t_3,t_peak,v_peak,t_6
    return v

# INFECTIOUSNESS FUNCTIONS


def proportional(v,cutoff=6):
    '''
    proportional(v,cutoff=6)
        returns the infectiousness of a viral load v
        for the proportional model.
    '''
    if v < cutoff:
        return 0
    else:
        x = 10**(v-cutoff)
        return x


def threshold(v,cutoff=6):
    '''
    threshold(v,cutoff=6)
        returns the infectiousness of a viral load v
        for the threshold model.
    '''
    if v < cutoff:
        return 0
    else:
        return 1


def logproportional(v,cutoff=6):
    '''
    logproportional(v,cutoff=6)
        returns the infectiousness of a viral load v
        for the logproportional model
        (Manuscript default)
    '''
    if v < cutoff:
        return 0
    else:
        return v-cutoff


# INFECTIOUSNESS REMOVED SAMPLING


def infectiousness_removed_indiv(D,L,inf,asymptomatic=0.65,dt=0,cutoff=6,se=1):
    '''
    infectiousness_removed_indiv_symptomatic(D,L,inf,dt=0,cutoff=6)
        D: days between tests
        L: log10 limit of detection of test (PCR: 3, RT-LAMP: 5)
        inf: a function handle {proportional,logproportional,threshold}
        asymptomatic: fraction of individuals assumed be asymptomatic
        dt: fixed time delay to return results
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)
        se: per-test sensitivity, i.e. probabilty that the test doesn't just fail due to e.g. bad sampling

    Returns the amount of infectiousness (arbitrary units) removed by testing, symptoms, and total
    for an INDIVIDUAL trajectory drawn randomly from the model, probabilistically symptomatic or asymptomatic.
    '''
    if np.random.random()<asymptomatic:
        a,b,c = infectiousness_removed_indiv_asymptomatic(D,L,inf,dt=dt,cutoff=cutoff,se=se)
    else:
        a,b,c = infectiousness_removed_indiv_symptomatic(D,L,inf,dt=dt,cutoff=cutoff,se=se)
    return a,b,c


def infectiousness_removed_indiv_symptomatic(D,L,inf,dt=0,cutoff=6,se=1):
    '''
    infectiousness_removed_indiv_symptomatic(D,L,inf,dt=0,cutoff=6,se=1)
        D: days between tests
        L: log10 limit of detection of test (PCR: 3, RT-LAMP: 5)
        inf: a function handle {proportional,logproportional,threshold}
        dt: fixed time delay to return results
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)
        se: per-test sensitivity, i.e. probabilty that the test doesn't just fail due to e.g. bad sampling

    Returns the amount of infectiousness (arbitrary units) removed by testing, symptoms, and total
    for an INDIVIDUAL symptomatic trajectory drawn randomly from the model.
    '''
    V,t_symptoms = get_trajectory(is_symptomatic=True)
    I = [inf(x,cutoff=cutoff) for x in V]
    total = np.sum(I)
    phase = np.random.choice(D)
    removed_by_testing = 0
    removed_by_symptoms = 0

    t_test = np.arange(phase,28,D)
    t_test = np.sort(np.random.choice(t_test,size=np.random.binomial(len(t_test),se),replace=False))
    t_pos = t_test[np.where(V[t_test]>L)[0]]
    if len(t_pos)==0:
        removed_by_testing = 0
        removed_by_symptoms = np.sum(I[t_symptoms:])
    else:
        t_dx = t_pos[0]+dt
        if t_dx == t_symptoms:
            # break ties with a coin flip
            if np.random.rand()<0.5:
                removed_by_testing = np.sum(I[t_dx:])
                removed_by_symptoms = 0
            else:
                removed_by_testing = 0
                removed_by_symptoms = np.sum(I[t_dx:])
        elif t_dx < t_symptoms:
            removed_by_testing = np.sum(I[t_dx:])
            removed_by_symptoms = 0
        elif t_dx > t_symptoms:
            removed_by_testing = 0
            removed_by_symptoms = np.sum(I[t_symptoms:])
    return removed_by_testing,removed_by_symptoms,total


def infectiousness_removed_indiv_asymptomatic(D,L,inf,dt=0,cutoff=6,se=1):
    '''
    infectiousness_removed_indiv_asymptomatic(D,L,inf,dt=0,cutoff=6,se=1)
        D: days between tests
        L: log10 limit of detection of test (PCR: 3, RT-LAMP: 5)
        inf: a function handle {proportional,logproportional,threshold}
        dt: fixed time delay to return results
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)
        se: per-test sensitivity, i.e. probabilty that the test doesn't just fail due to e.g. bad sampling

    Returns the amount of infectiousness (arbitrary units) removed by testing, symptoms, and total
    for an INDIVIDUAL asymptomatic trajectory drawn randomly from the model.
    '''
    V = get_trajectory(is_symptomatic=False)
    I = [inf(x,cutoff=cutoff) for x in V]
    total = np.sum(I)
    phase = np.random.choice(D)
    removed_by_testing = 0
    removed_by_symptoms = 0

    t_test = np.arange(phase,28,D)
    t_test = np.sort(np.random.choice(t_test,size=np.random.binomial(len(t_test),se),replace=False))
    t_pos = t_test[np.where(V[t_test]>L)[0]]
    if len(t_pos)==0:
        removed_by_testing = 0
    else:
        t_dx = t_pos[0]+dt
        removed_by_testing = np.sum(I[t_dx:])
        removed_by_symptoms = 0
    return removed_by_testing,removed_by_symptoms,total


def infectiousness_removed_pop(D,L,inf,asymptomatic=0.65,dt=0,cutoff=6,n_samples=1000,se=1):
    '''
    infectiousness_removed_pop(D,L,inf,asymptomatic,dt=0,cutoff=6,n_samples=1000,se=1)
        D: days between tests
        L: log10 limit of detection of test (PCR: 3, RT-LAMP: 5)
        inf: a function handle {proportional,logproportional,threshold}
        asymptomatic: fraction of individuals assumed be asymptomatic
        dt: fixed time delay to return results
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)
        n_samples: the number of individuals to sample in computing the population
        se: per-test sensitivity, i.e. probabilty that the test doesn't just fail due to e.g. bad sampling

    Returns the average amount of infectiousness (arbitrary units) removed by testing, symptoms, and total
    for n_samples individual trajectories drawn randomly from the model, with a stochastic proportion asymptomatic.
    '''
    test,self,total = 0,0,0
    for i in range(n_samples):
        a,b,c = infectiousness_removed_indiv(D,L,inf,asymptomatic=asymptomatic,dt=dt,cutoff=cutoff,se=se)
        test+=a
        self+=b
        total+=c
    return test/n_samples,self/n_samples,total/n_samples


def get_R_reduction_factor(D,L,inf,asymptomatic=0.65,dt=0,cutoff=6,n_samples=10000,se=1):
    '''
    get_R_reduction_factor(D,L,inf,asymptomatic=0.65,dt=0,cutoff=6,n_samples=10000,se=1)
        D: days between tests
        L: log10 limit of detection of test (PCR: 3, RT-LAMP: 5)
        inf: a function handle {proportional,logproportional,threshold}
        asymptomatic: fraction of individuals assumed be asymptomatic
        dt: fixed time delay to return results
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)
        n_samples: the number of individuals to sample in computing the population
        se: per-test sensitivity, i.e. probabilty that the test doesn't just fail due to e.g. bad sampling

    Returns the factor by which R is likely to be reduced via a particular policy.
    '''
    pol,no_pol = 0,0
    for i in range(n_samples):
        a,b,c = infectiousness_removed_indiv(D,L,inf,asymptomatic=asymptomatic,dt=dt,cutoff=cutoff,se=se)
        no_pol += (c-b)
        pol += (c-(a+b))
    return pol/no_pol


# SEIR FULLY MIXED MODEL


def compute_factor_to_calibrate_R0_SQ(infectiousness,asymptomatic,cutoff):
    '''
    compute_factor_to_calibrate_R0_SQ(infectiousness,asymptomatic,cutoff)
        infectiousness: a function handle {proportional,logproportional,threshold}
        asymptomatic: the fraction of individuals who are asymptomatic [0,1]
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)

    Returns a constant mean_infectiousness which is used to scale absolute infectiousness in simulation
    '''
    total_infectiousness = 0
    n_draws = 10000
    for i in range(n_draws):
        if np.random.random() < asymptomatic:
            VL = get_trajectory(is_symptomatic=False)
            IN = [infectiousness(x,cutoff=cutoff) for x in VL]
            total_infectiousness += np.sum(IN)
        else:
            VL,t_symptoms = get_trajectory(is_symptomatic=True)
            IN = [infectiousness(x,cutoff=cutoff) for x in VL]
            total_infectiousness += np.sum(IN[:t_symptoms])
    mean_infectiousness = total_infectiousness/n_draws
    return mean_infectiousness


def get_Reff(N,D,L,infectiousness_function,asymptomatic=0.65,results_delay=0,R0=2.5,cutoff=6):
    '''
    get_Reff(N,D,L,infectiousness_function,asymptomatic=0.65,results_delay=0,R0=2.5,cutoff=6)
        N: population size
        D: days between tests
        L: log10 limit of detection of test (PCR: 3, RT-LAMP: 5)
        infectiousness_function: a function handle {proportional,logproportional,threshold}
        asymptomatic: fraction asymptomatic
        results_delay: fixed time delay to return results
        R0: basic reproductive number
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)
    '''
    I_init=int(0.005*N)
    _,_,_,_,_,_,Iint = SEIRsimulation(
        N=N,
        external_rate=0,
        D=D,
        L=L,
        infectiousness_function=infectiousness_function,
        asymptomatic=asymptomatic,
        results_delay=results_delay,
        R0=R0,
        cutoff=cutoff,
        I_init=I_init,
        tmax=15,
        calibration_mode=True
        )
    return Iint/I_init


def SEIRsimulation(N,external_rate,D,L,infectiousness_function,asymptomatic=0.65,results_delay=0,R0=2.5,cutoff=6,I_init=0,tmax=365,calibration_mode=False):
    '''
    get_Reff(N,external_rate,D,L,infectiousness_function,asymptomatic=0.65,results_delay=0,R0=2.5,cutoff=6,I_init=0,tmax=365,calibration_mode=False)
        N: population size
        external_rate: IID probability that each su gets infected each day
        D: days between tests
        L: log10 limit of detection of test (PCR: 3, RT-LAMP: 5)
        infectiousness_function: a function handle {proportional,logproportional,threshold}
        asymptomatic: fraction asymptomatic
        results_delay: fixed time delay to return results
        R0: basic reproductive number
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)
        I_init: number of individuals initially infected (all others will be initially susceptible)
        tmax: duration of simulation (days)
        calibration_mode: False runs simulations as normal. True places internal infections immediately into quarantine to prevent secondary internal transmission and more accurately compute R0
    Returns:
        St,It,Rt,Qt,SQt,np.sum(was_infected_ext),np.sum(was_infected_int)
        St: timeseries of the number of individuals in susceptible group on each day
        It: timeseries of the number of individuals in infected group on each day
        Rt: timeseries of the number of individuals in recovered group on each day
        Qt: timeseries of the number of individuals in quarantined group on each day
        SQt: timeseries of the number of individuals in self-quarantined group on each day
        Total number of external infections
        Total number of internal infections i.e. total infections minus external infections
    '''
    c = compute_factor_to_calibrate_R0_SQ(infectiousness_function,asymptomatic,cutoff)
    k = R0/(c*(N-1))
    
    was_infected_ext = np.zeros(N)
    was_infected_int = np.zeros(N)
    
    is_symptomatic = np.random.binomial(1,1-asymptomatic,size=N)
    t_symptoms = np.zeros(N)
    viral_loads = {}
    infectious_loads = {}
    for i in range(N):
        if is_symptomatic[i]:
            viral_loads[i],t_symptoms[i] = get_trajectory(is_symptomatic=True)
        else:
            viral_loads[i] = get_trajectory(is_symptomatic=False)
        infectious_loads[i] = [infectiousness_function(x,cutoff=cutoff)*k for x in viral_loads[i]]

    test_schedule = np.random.choice(D,size=N)

    S = set(np.arange(N))
    I = set()
    R = set()
    Q = set()
    SQ = set()
    St = []
    It = []
    Rt = []
    Qt = []
    SQt = []

    infection_day = np.zeros(N,dtype=int)
    days_till_results = -1*np.ones(N)

    for t in range(tmax):
        # Test
        tested = np.where((test_schedule + t) % D==0)[0]
        for i in tested:
            if (i in Q) or (i in R) or (i in SQ):
                continue
            if days_till_results[i] > -1:
                continue
            if viral_loads[i][infection_day[i]] > L:
                days_till_results[i] = results_delay

        # Isolate
        for i in I:
            if days_till_results[i]==0:
                I = I - set([i])
                Q = Q.union(set([i]))
        
        # Self Quarantine
        for i in I:
            if is_symptomatic[i]==1:
                if infection_day[i]==t_symptoms[i]:
                    I = I-set([i])
                    SQ = SQ.union(set([i]))

        # Initial infections
        if t==0:
            initial_infections = np.random.choice(list(S),size=I_init,replace=False)
            was_infected_ext[initial_infections] = 1    
            S = S - set(initial_infections)
            I = I.union(set(initial_infections))

        # External infections
        ext_infections = np.random.choice(list(S),np.random.binomial(len(S),external_rate))
        was_infected_ext[ext_infections] = 1
        S = S - set(ext_infections)
        I = I.union(set(ext_infections))

        # Internal infections
        infectiousnesses = [infectious_loads[i][infection_day[i]] for i in I]
        p_int_infection = 1 - np.prod(1-np.array(infectiousnesses))
        # print(len(I),p_int_infection)
        int_infections = np.random.choice(list(S),np.random.binomial(len(S),p_int_infection))
        was_infected_int[int_infections] = 1
        S = S - set(int_infections)
        if calibration_mode==False:
            I = I.union(set(int_infections))
        else:
            Q = Q.union(set(int_infections))

        # Update all infection days
        for i in I:
            if (infection_day[i] == 27) or ((viral_loads[i][infection_day[i]]<6) and (infection_day[i]>7)):
                I = I - set([i])
                R = R.union(set([i]))
            infection_day[i] += 1
        for q in Q:
            if (infection_day[q] == 27) or ((viral_loads[q][infection_day[q]]<6) and (infection_day[q]>7)):
                Q = Q - set([q])
                R = R.union(set([q]))
            infection_day[q] += 1
        for q in SQ:
            if (infection_day[q] == 27) or ((viral_loads[q][infection_day[q]]<6) and (infection_day[q]>7)):
                SQ = SQ - set([q])
                R = R.union(set([q]))
            infection_day[q] += 1
        
        # Update the results delays:
        for i in range(N):
            if days_till_results[i] > -1:
                days_till_results[i] = days_till_results[i] - 1
        
        St.append(len(S))
        It.append(len(I))
        Rt.append(len(R))
        Qt.append(len(Q))
        SQt.append(len(SQ))
    return St,It,Rt,Qt,SQt,np.sum(was_infected_ext),np.sum(was_infected_int)


def SEIRsimulation_suppression(N,
                               external_rate,
                               D,
                               L,
                               infectiousness_function,
                               prev_cutoff,
                               se_sample,
                               refusal_rate,
                               asymptomatic=0.65,
                               results_delay=0,
                               R0=2.5,
                               cutoff=6,
                               I_init=0,
                               tmax=365,
                               calibration_mode=False):
    '''
    get_Reff(N,external_rate,D,L,infectiousness_function,asymptomatic=0.65,results_delay=0,R0=2.5,cutoff=6,I_init=0,tmax=365,calibration_mode=False)
        *** indicates parameters added to this model vs the SEIRsimulation model
        N: population size
        external_rate: IID probability that each su gets infected each day
        D: days between tests
        L: log10 limit of detection of test (PCR: 3, RT-LAMP: 5)
        infectiousness_function: a function handle {proportional,logproportional,threshold}
        *** prev_cutoff: prevalence value at which testing will be initiated
        *** se_sample: sample collection sensitivity, i.e. probability that the test sample was collected without error
        *** refusal_rate: fraction of individuals who will always refuse testing
        asymptomatic: fraction asymptomatic
        results_delay: fixed time delay to return results
        R0: basic reproductive number
        cutoff: the minimum value of log10 viral load for infectiousness. (Manuscript default: 6)
        I_init: number of individuals initially infected (all others will be initially susceptible)
        tmax: duration of simulation (days)
        calibration_mode: False runs simulations as normal. True places internal infections immediately into quarantine to prevent secondary internal transmission and more accurately compute R0
    Returns:
        return St,It,Rt,Qt,SQt,np.sum(was_infected_ext),np.sum(was_infected_int),pre_control_external,pre_control_internal
        St: timeseries of the number of individuals in susceptible group on each day
        It: timeseries of the number of individuals in infected group on each day
        Rt: timeseries of the number of individuals in recovered group on each day
        Qt: timeseries of the number of individuals in quarantined group on each day
        SQt: timeseries of the number of individuals in self-quarantined group on each day
        Total number of external infections
        Total number of internal infections i.e. total infections minus external infections
        Total number of external infections prior to suppression testing
        Total number of internal infections prior to suppression testing
    '''
    is_control_on = False
    c = compute_factor_to_calibrate_R0_SQ(infectiousness_function,asymptomatic,cutoff)
    k = R0/(c*(N-1))
    
    was_infected_ext = np.zeros(N)
    was_infected_int = np.zeros(N)
    
    is_symptomatic = np.random.binomial(1,1-asymptomatic,size=N)
    is_refuser = np.random.binomial(1,refusal_rate,size=N)
    t_symptoms = np.zeros(N)
    viral_loads = {}
    infectious_loads = {}
    for i in range(N):
        if is_symptomatic[i]:
            viral_loads[i],t_symptoms[i] = get_trajectory(is_symptomatic=True)
        else:
            viral_loads[i] = get_trajectory(is_symptomatic=False)
        infectious_loads[i] = [infectiousness_function(x,cutoff=cutoff)*k for x in viral_loads[i]]

    test_schedule = np.random.choice(D,size=N)

    S = set(np.arange(N))
    I = set()
    R = set()
    Q = set()
    SQ = set()
    St = []
    It = []
    Rt = []
    Qt = []
    SQt = []

    infection_day = np.zeros(N,dtype=int)
    days_till_results = -1*np.ones(N)

    for t in range(tmax):
        if (len(I)/N > prev_cutoff) and (is_control_on==False):
            is_control_on=True
            pre_control_internal = np.sum(was_infected_int)
            pre_control_external = np.sum(was_infected_ext)
        
        if is_control_on==True:
            # Test
            tested = np.where((test_schedule + t) % D==0)[0]
            for i in tested:
                if not is_refuser[i]:
                    if (i in Q) or (i in R) or (i in SQ):
                        continue
                    if days_till_results[i] > -1:
                        continue
                    if viral_loads[i][infection_day[i]] > L:
                        if np.random.random() < se_sample:
                            days_till_results[i] = results_delay
                        else:
                            continue

        # Isolate
        for i in I:
            if days_till_results[i]==0:
                I = I - set([i])
                Q = Q.union(set([i]))
        
        # Self Quarantine
        for i in I:
            if is_symptomatic[i]==1:
                if infection_day[i]==t_symptoms[i]:
                    I = I-set([i])
                    SQ = SQ.union(set([i]))

        # Initial infections
        if t==0:
            initial_infections = np.random.choice(list(S),size=I_init,replace=False)
            was_infected_ext[initial_infections] = 1    
            S = S - set(initial_infections)
            I = I.union(set(initial_infections))

        # External infections
        ext_infections = np.random.choice(list(S),np.random.binomial(len(S),external_rate))
        was_infected_ext[ext_infections] = 1
        S = S - set(ext_infections)
        I = I.union(set(ext_infections))

        # Internal infections
        infectiousnesses = [infectious_loads[i][infection_day[i]] for i in I]
        p_int_infection = 1 - np.prod(1-np.array(infectiousnesses))
        # print(len(I),p_int_infection)
        int_infections = np.random.choice(list(S),np.random.binomial(len(S),p_int_infection))
        was_infected_int[int_infections] = 1
        S = S - set(int_infections)
        if calibration_mode==False:
            I = I.union(set(int_infections))
        else:
            Q = Q.union(set(int_infections))

        # Update all infection days
        for i in I:
            if (infection_day[i] == 27) or ((viral_loads[i][infection_day[i]]<6) and (infection_day[i]>7)):
                I = I - set([i])
                R = R.union(set([i]))
            infection_day[i] += 1
        for q in Q:
            if (infection_day[q] == 27) or ((viral_loads[q][infection_day[q]]<6) and (infection_day[q]>7)):
                Q = Q - set([q])
                R = R.union(set([q]))
            infection_day[q] += 1
        for q in SQ:
            if (infection_day[q] == 27) or ((viral_loads[q][infection_day[q]]<6) and (infection_day[q]>7)):
                SQ = SQ - set([q])
                R = R.union(set([q]))
            infection_day[q] += 1
        
        # Update the results delays:
        for i in range(N):
            if days_till_results[i] > -1:
                days_till_results[i] = days_till_results[i] - 1
        
        St.append(len(S))
        It.append(len(I))
        Rt.append(len(R))
        Qt.append(len(Q))
        SQt.append(len(SQ))
    return St,It,Rt,Qt,SQt,np.sum(was_infected_ext),np.sum(was_infected_int),pre_control_external,pre_control_internal

