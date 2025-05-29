SELECT 
	conference.name 
FROM 
	author, organization, writes, publication, conference, domain_conference, domain 
WHERE
	author.oid = organization.oid
	AND author.aid = writes.aid
    AND writes.pid = publication.pid
    AND publication.cid = conference.cid
    AND conference.cid = domain_conference.cid
    AND domain_conference.did = domain.did
    AND author.paper_count > 10
    AND organization.name = 'Tel Aviv University'
    AND publication.year > 2010
    AND publication.citation_count > 1
GROUP BY conference.name 