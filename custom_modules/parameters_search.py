from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# подбор наилучших параметров по определенной метрике
class RandomSearch(object):
    
    # выдает лучшие параметры и их  для данного вида моделей (быстрее, чем грид серч)
    # scoring: чем выше значение, тем лучше (не по модулю)
    def __init__(self,X_train,y_train,model,hyperparameters, n_iter=20, scoring = 'neg_mean_absolute_error', sample_weight = None):
        
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.hyperparameters = hyperparameters
        # Create grid search 10-fold cross validation and 100 iterations
        cv = 5
        regr = RandomizedSearchCV(self.model,
                                 self.hyperparameters,
                                 random_state=1,
                                 n_iter=n_iter, # количество разных комбинаций
                                 cv=cv, # кол-во фолдов в кросс-валидации
                                 verbose=2, # степень детализации сообщений о каждой комбинации
                                 #n_jobs=-1,
                                 scoring = scoring # способ оценки качества
                                 )
        
        # если есть специфические веса важности у каждого объекта
        if sample_weight is not None: 
            best_model = regr.fit(self.X_train, self.y_train, sample_weight = sample_weight)
        else:
            best_model = regr.fit(self.X_train, self.y_train)
        message = (best_model.best_score_, best_model.best_params_) # выдает лучшие параметры анализируемой модели
        print("Best: %f using %s" % (message))
        self.model = best_model
    
    def BestModelPridict(self,X_test):
        
        best_model,_ = self.RandomSearch() # инициализируем наш класс и находим лучшие параметры (они хранятся в self)
        pred = best_model.predict(X_test)
        return pred


# то же, но для грид серча
class GridSearch(object):
    
    def __init__(self,X_train,y_train,model,hyperparameters, scoring = 'neg_mean_absolute_error', sample_weight = None):
        
        self.X_train = X_train
        self.y_train = y_train
        self.model = model
        self.hyperparameters = hyperparameters
        # Create randomized search 10-fold cross validation and 100 iterations
        cv = 5
        regr = GridSearchCV(self.model,
                                 self.hyperparameters,
                                 cv=cv,
                                 verbose=2,
                                 #n_jobs=-1,
                                 scoring = scoring
                                 )
        # Fit randomized search
        if sample_weight is not None:
            best_model = regr.fit(self.X_train, self.y_train, sample_weight = sample_weight)
        else:
            best_model = regr.fit(self.X_train, self.y_train)
        message = (best_model.best_score_, best_model.best_params_) 
        print("Best: %f using %s" % (message))
        self.model = best_model
    
    def BestModelPridict(self,X_test):
        
        best_model,_ = self.GridSearch()
        pred = best_model.predict(X_test)
        return pred