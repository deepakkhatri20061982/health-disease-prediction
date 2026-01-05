pipeline {
    agent any
    
    stages {
        stage('Force Clean Workspace') {
            steps {
                script {
                    def ws = pwd()
                    bat """
                        echo Cleaning workspace: %ws%
                        rmdir /s /q "%ws%" || exit 0
                        mkdir "%ws%"
                    """
                }
            }
        }
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
                sh 'python health_disease_model_training_v2.py'
            }
        }
        stage('Run Unit Tests') {
            steps {
                sh '''
                    export PYTHONPATH=$WORKSPACE:$PYTHONPATH && pytest -v --disable-warnings --maxfail=1 --cov=health_disease_model_training --cov-report=xml --cov-report=term
                '''
            }
        }
    }   
}
