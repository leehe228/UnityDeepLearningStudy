using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using PA_DronePack;

public class Drone : MonoBehaviour
{
    public GameObject s;
    public int catchNum;
    public Vector3 boxPos;
    public LineRenderer line;

    // Start is called before the first frame update
    void Start()
    {
        catchNum = 0;
        line = GetComponent<LineRenderer>();
        line.startWidth = 0.05f;
        line.endWidth = 0.05f;
        line.SetPosition(0, new Vector3(0f, -10f, 0f));
        line.SetPosition(1, new Vector3(0f, -10f, 0f));
    }

    // Update is called once per frame
    void Update()
    {
        if (catchNum > 0) {
            if (s.transform.position.y < 1f) {
                s.transform.position = new Vector3(s.transform.position.x, 1f, s.transform.position.z);
            }
            line.SetPosition(0, s.transform.position);
            line.SetPosition(1, boxPos);
        } 
        else {
            line.SetPosition(0, new Vector3(0f, -10f, 0f));
            line.SetPosition(1, new Vector3(0f, -10f, 0f));
        }
    }

    void OnCollisionEnter(Collision other) {
        if (other.gameObject.CompareTag("agent")) {
            gameObject.GetComponent<DroneAgent>().GiveReward(-0.1f);
        }    
    }
}
