class DataScientist():
    '''This class describes Data Science projects I realized.
    '''

    def __init__(self, github='Jennifer974', school='VIVADATA'):
        '''This function initializes attributes of DataScientist class
        '''
        self.github = github        #Name of my Github account
        self.school = school        #Name of my school

    def data_scientist_projects(self):
        '''This function prints desciption of Data Science projects I realized.

        Parameters
        --------------

        Returns
        --------------
        Projects description
        '''
        #Projects Dictionary
        projects_dict = {'University of Oxford': 'Voice identification',
                         'Kaggle':
                         {'Anomaly detection': 'Credit card fraud detection',
                          'NLP': 'Quora insincere questions classification',
                          'Churn': 'Telcom Customer Churn'}}

        print('I\'m a Data Scientist at', self.school, '(Github :', self.github, ') \nI realize this following projects :')
        for project, description in projects_dict.items():
            print('-', project, ':', description)

def main():
    '''This function is defined to print class output.
    '''
    #Instanciate DataScientist class
    my_profile = DataScientist()

    #Execute data_scientist_projects function
    my_profile.data_scientist_projects()

if __name__ == "__main__":
    main()
