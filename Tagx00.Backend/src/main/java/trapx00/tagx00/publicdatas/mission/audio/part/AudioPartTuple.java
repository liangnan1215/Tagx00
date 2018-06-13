package trapx00.tagx00.publicdatas.mission.audio.part;


import trapx00.tagx00.publicdatas.mission.TagDescriptionTuple;

public class AudioPartTuple {
    private double startOffset;
    private double endOffset;
    private TagDescriptionTuple tuple;

    public AudioPartTuple(double startOffset, double endOffset, TagDescriptionTuple tuple) {
        this.startOffset = startOffset;
        this.endOffset = endOffset;
        this.tuple = tuple;
    }

    public AudioPartTuple() {
    }

    public double getStartOffset() {
        return startOffset;
    }

    public void setStartOffset(double startOffset) {
        this.startOffset = startOffset;
    }

    public double getEndOffset() {
        return endOffset;
    }

    public void setEndOffset(double endOffset) {
        this.endOffset = endOffset;
    }

    public TagDescriptionTuple getTuple() {
        return tuple;
    }

    public void setTuple(TagDescriptionTuple tuple) {
        this.tuple = tuple;
    }
}
