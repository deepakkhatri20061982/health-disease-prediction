pipeline {
    agent any
    
    stages {
        stage("Install Dependencies") {
            steps {
                sh """
                pip install --upgrade pip
                pip install -r requirements.txt
                """
            }
        }
        stage('Verify Python Version') {
            steps {
                sh '''
                python --version
                which python
                pip --version
                '''
            }
        }
        stage('Model Training') {
            steps {
                sh 'python src/eda_Model_Updated_DK.py'
            }
        }
    }   
}
