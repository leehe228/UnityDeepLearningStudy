using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using MLAgents;

namespace PA_DronePack
{
    public class DroneAgent : Agent
    {

        private PA_DroneController dcoScript;
        public GameObject goal;
        Vector3 droneInitPos;
        Quaternion droneInitRot;
        float preDist, curDist;
        

        // obs
        public float isCatch = 0f;
        public float boxType = -1f;
        public GameObject map;

        public override void InitializeAgent()
        {
            dcoScript = gameObject.GetComponent<PA_DroneController>();
            droneInitPos = gameObject.transform.position;
            droneInitRot = gameObject.transform.rotation;
        }

        public override void CollectObservations()
        {
            AddVectorObs(isCatch);
            AddVectorObs(boxType);
            AddVectorObs(map.GetComponent<MapController>().dest1.transform.position);
            AddVectorObs(map.GetComponent<MapController>().dest2.transform.position);
            AddVectorObs(map.GetComponent<MapController>().box1.transform.position);
            AddVectorObs(map.GetComponent<MapController>().box2.transform.position);
            AddVectorObs(gameObject.transform.position);
            
            string thisName = gameObject.name;
            GameObject[] agents = GameObject.FindGameObjectsWithTag("agent");

            foreach (GameObject a in agents) {
                Debug.Log(a.name);
                if (a.name != thisName) {
                    AddVectorObs(a.transform.position);
                }
            }

            AddVectorObs(gameObject.GetComponent<Rigidbody>().velocity);
            AddVectorObs(gameObject.GetComponent<Rigidbody>().angularVelocity);
        }

        public override void AgentAction(float[] vectorAction, string textAction)
        {
            var act0 = Mathf.Clamp(vectorAction[0], -1f, 1f);
            var act1 = Mathf.Clamp(vectorAction[1], -1f, 1f);
            var act2 = Mathf.Clamp(vectorAction[2], -1f, 1f);

            dcoScript.DriveInput(act0);
            dcoScript.StrafeInput(act1);
            dcoScript.LiftInput(act2);

            /*if ((goal.transform.position - gameObject.transform.position).magnitude < 0.5f)
            {
                SetReward(1);
                Done();
            }*/
            /*else if ((goal.transform.position - gameObject.transform.position).magnitude > 10f)
            {
                SetReward(-1);
                Done();
            }
            else*/
            /*{
                curDist = (goal.transform.position - gameObject.transform.position).magnitude;
                var reward = (preDist - curDist);
                SetReward(reward);
                preDist = curDist;
            }*/
        }

        public void GiveReward(float r) {
            Debug.Log("Reward Recieved.");
            SetReward(r);
        }

        public void GiveDone() {
            Done();
        }

        public override void AgentReset()
        {
            /*gameObject.GetComponent<Rigidbody>().velocity = Vector3.zero;
            gameObject.GetComponent<Rigidbody>().angularVelocity = Vector3.zero;
            gameObject.transform.position = droneInitPos;
            gameObject.transform.rotation = droneInitRot;

            goal.transform.position = gameObject.transform.position 
                + new Vector3(Random.Range(-10f, 10f), Random.Range(1f, 3f), Random.Range(-10f, 10f));
            preDist = (goal.transform.position - gameObject.transform.position).magnitude;*/
        }

        public override void AgentOnDone()
        {

        }
        
        void OnTriggerEnter(Collider col) {
            //SetReward(-1);
            //Done();
        }

        void OnTriggerStay(Collider col) {
            //SetReward(-0.05f);
        }

        void OnCollisionEnter(Collision other) {
            /*if (other.gameObject.CompareTag("box1")) {
                other.gameObject.SetActive(false);
            }*/
        }
    }
}

