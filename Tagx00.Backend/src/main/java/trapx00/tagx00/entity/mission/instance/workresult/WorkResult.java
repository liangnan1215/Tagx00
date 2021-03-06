package trapx00.tagx00.entity.mission.instance.workresult;

import javax.persistence.Embeddable;
import java.io.Serializable;

@Embeddable
public class WorkResult implements Serializable {
    private String workResultId;
    private boolean isDone;

    public WorkResult() {
    }

    public WorkResult(String workResultId, boolean isDone) {
        this.workResultId = workResultId;
        this.isDone = isDone;
    }

    public String getWorkResultId() {
        return workResultId;
    }

    public void setWorkResultId(String workResultId) {
        this.workResultId = workResultId;
    }

    public boolean isDone() {
        return isDone;
    }

    public void setDone(boolean isDone) {
        this.isDone = isDone;
    }
}
