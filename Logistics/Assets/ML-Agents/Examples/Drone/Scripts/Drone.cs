using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using PA_DronePack;

public class Drone : MonoBehaviour
{
    public int catchNum;
    public int boxType;
    public Vector3 boxPos;
    public LineRenderer line;

    void Start()
    {
        catchNum = 0;
        line = GetComponent<LineRenderer>();
        line.startWidth = 0.05f;
        line.endWidth = 0.05f;
        line.SetPosition(0, new Vector3(0f, -10f, 0f));
        line.SetPosition(1, new Vector3(0f, -10f, 0f));
    }

    void Update()
    {
        if (catchNum > 0) {
            if (gameObject.transform.position.y < 1f) {
                gameObject.transform.position = new Vector3(gameObject.transform.position.x, 1f, gameObject.transform.position.z);
            }
            line.SetPosition(0, gameObject.transform.position);
            line.SetPosition(1, boxPos);
        } 
        else {
            line.SetPosition(0, new Vector3(0f, -10f, 0f));
            line.SetPosition(1, new Vector3(0f, -10f, 0f));
        }
    }

    void OnCollisionEnter(Collision other) {
        if (other.gameObject.CompareTag("agent")) {
            // collide with another agent
            gameObject.GetComponent<DroneAgent>().GiveReward(-0.2f);
        }
        else if (other.gameObject.CompareTag("map")) {
            // collide with wall
            gameObject.GetComponent<DroneAgent>().GiveReward(-0.2f);
        }
    }
}
