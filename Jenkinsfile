pipeline {
    agent {
        docker {
            image 'python:3.10'
        }
    }

    stages {
        stage("Install Dependencies") {
            steps {
                sh """
                pip install --upgrade pip
                pip install -r requirements.txt
                """
            }
        }
    }
}
