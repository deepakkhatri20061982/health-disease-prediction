pipeline {
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
    }   
}
