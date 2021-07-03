using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MapController : MonoBehaviour
{

    public GameObject box1, box2, box3;
    public GameObject dest1, dest2, dest3;

    // Start is called before the first frame update
    void Start()
    {
        generate1();
        generate2();
        // generate3();
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void generate1() {
        box1.transform.position = new Vector3(Random.Range(-2f, 2f), 3f, Random.Range(-2f, 2f));

        int rx1 = Random.Range(0, 2);
        if (rx1 == 0) rx1 = -1;

        int rz1 = Random.Range(0, 2);
        if (rz1 == 0) rz1 = -1;

        dest1.transform.position = new Vector3(rx1 * Random.Range(4f, 8f), -0.45f, rz1 * Random.Range(4f, 8f));

        //Invoke("generate", 3);
    }

    public void generate2() {
        box2.transform.position = new Vector3(Random.Range(-2f, 2f), 3f, Random.Range(-2f, 2f));

        int rx2 = Random.Range(0, 2);
        if (rx2 == 0) rx2 = -1;

        int rz2 = Random.Range(0, 2);
        if (rz2 == 0) rz2 = -1;

        dest2.transform.position = new Vector3(rx2 * Random.Range(4f, 8f), -0.45f, rz2 * Random.Range(4f, 8f));

        //Invoke("generate", 3);
    }

    void generate3() {
        //Vector3 pos3 = new Vector3(Random.Range(-2f, 2f), 3f, Random.Range(-2f, 2f));

        //Instantiate(box3, pos3, Quaternion.identity);

        //int rx3 = Random.Range(0, 2);
        //if (rx3 == 0) rx3 = -1;

        //int rz3 = Random.Range(0, 2);
        //if (rz3 == 0) rz3 = -1;

        //Vector3 dpos3 = new Vector3(rx3 * Random.Range(4f, 8f), -0.45f, rz3 * Random.Range(4f, 8f));

        // dest3.transform.position = dpos3;

        //Invoke("generate", 3);
    }
}
